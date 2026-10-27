package server

import (
	"context"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cluster"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

// clusterLocalFitPercent is how much of the local memory a model may use
// before the scheduler brings in a cluster worker.
const clusterLocalFitPercent = 90

// selectModelPlacement is selectLlamaServerPlacementWithReclaim for a named
// model, so placement rules for single models apply.
func selectModelPlacement(model string, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, predictedVRAM, reclaimable uint64, opts api.Options) ([]ml.DeviceInfo, api.Options) {
	if local, remote := cluster.SplitDevices(gpus); len(remote) > 0 {
		return selectClusterPlacement(model, systemInfo, local, remote, predictedVRAM, reclaimable, opts)
	}
	selected, launchOpts := selectLlamaServerPlacementWithReclaim(systemInfo, gpus, predictedVRAM, reclaimable, opts)
	// memory kept free here holds also while no worker is usable
	if cfg, err := cluster.LoadConfig(); err == nil && cfg.KeepFreeHere > 0 && len(cfg.ActiveWorkers()) > 0 && opts.NumGPU != 0 {
		selected = withFitMargin(selected, cfg.KeepFreeHere)
	}
	return selected, launchOpts
}

// workersOnly reports whether a placement uses cluster workers and no device
// of this machine.
func workersOnly(gpus []ml.DeviceInfo) bool {
	local, remote := cluster.SplitDevices(gpus)
	return len(local) == 0 && len(remote) > 0
}

// workerMargin is memory left free on a worker when the model is fitted
// there; it matches the llama-server default fit target.
const workerMargin = 1024 * format.MebiByte

// selectClusterPlacement places a model when cluster worker devices are
// available. Under the default policy a worker is used only when the model
// does not fit on this machine, counting memory that unloading other Ollama
// models would free; small models never pay for the trip over the link and
// the worker stays free for the models that need it. The cluster config can
// keep memory free on this machine, send models from a size up to the worker
// first (worker-first) and fix the placement of single models.
func selectClusterPlacement(model string, systemInfo ml.SystemInfo, local, remote []ml.DeviceInfo, predictedVRAM, reclaimable uint64, opts api.Options) ([]ml.DeviceInfo, api.Options) {
	cfg, err := cluster.LoadConfig()
	if err != nil {
		cfg = &cluster.Config{}
	}
	keepFree := cfg.KeepFreeHere

	localOnly := func(reason string) ([]ml.DeviceInfo, api.Options) {
		slog.Debug("keeping model on this machine", "model", model, "reason", reason)
		gpus, launchOpts := selectLlamaServerPlacementWithReclaim(systemInfo, local, predictedVRAM, 0, opts)
		return withFitMargin(gpus, keepFree), launchOpts
	}

	if opts.NumGPU == 0 || opts.MainGPU != nil {
		return localOnly("explicit num_gpu or main_gpu")
	}

	split := opts.ClusterSplit
	if split != nil && !*split {
		return localOnly("cluster_split=false")
	}

	place := cfg.Placement(model, predictedVRAM)
	if split != nil {
		place = cluster.PlaceSplit
	}
	if t := opts.ClusterTensor; t != nil {
		if *t {
			place = cluster.PlaceTensor
		} else if place == cluster.PlaceTensor {
			place = cfg.PolicyPlacement(predictedVRAM)
		}
	}
	if place == cluster.PlaceHere {
		return localOnly("placement rule")
	}

	group := bestLocalGroup(systemInfo, local)
	localAvailable, _, _ := availableMemoryForLoad(systemInfo, group)
	capacity := localAvailable + reclaimable
	capacity -= min(keepFree, capacity)

	if place == cluster.PlaceTensor {
		// without what unloading would free: the scheduler evicts by the
		// memory of all selected devices, so it would not make room here
		free := localAvailable - min(keepFree, localAvailable)
		selected, launchOpts, reason := selectTensorPlacement(cfg, group, remote, predictedVRAM, free, opts)
		if reason == "" {
			slog.Info("running model with tensor parallelism across this machine and the cluster worker", "model", model,
				"predicted", format.HumanBytes2(predictedVRAM),
				"local_available", format.HumanBytes2(free),
				"worker_available", format.HumanBytes2(selected[len(selected)-1].FreeMemory))
			logSelectedGPUGroup(append(local, remote...), selected)
			return selected, launchOpts
		}
		slog.Info("tensor parallelism not possible, placing the model by the policy", "model", model, "reason", reason)
		place = cfg.PolicyPlacement(predictedVRAM)
		if split != nil {
			place = cluster.PlaceSplit
		}
	}
	if place == cluster.PlaceAuto && (predictedVRAM == 0 || predictedVRAM <= capacity*clusterLocalFitPercent/100) {
		return localOnly("fits on this machine")
	}

	var workers []ml.DeviceInfo
	var workersFree uint64
	for _, d := range remote {
		if d.FreeMemory > 0 {
			workers = append(workers, d)
			workersFree += d.FreeMemory
		}
	}
	if len(workers) == 0 {
		return localOnly("cluster workers busy")
	}

	launchOpts := opts
	launchOpts.MainGPU = nil
	// an unknown size goes to the worker too; llama-server fits it there
	if place == cluster.PlaceWorker && predictedVRAM <= workersFree*clusterLocalFitPercent/100 {
		slog.Info("placing model on cluster workers", "model", model,
			"predicted", format.HumanBytes2(predictedVRAM),
			"workers_available", format.HumanBytes2(workersFree))
		logSelectedGPUGroup(append(local, remote...), workers)
		return slices.Clone(workers), launchOpts
	}

	margin := keepFree
	if place == cluster.PlaceWorker && predictedVRAM > 0 {
		// llama-server fills this machine first and gives the rest to the
		// workers; leave here only what the workers cannot take, plus a
		// tenth of the model for the estimate being off.
		var groupFree uint64
		for _, g := range group {
			groupFree += g.FreeMemory
		}
		onWorkers := workersFree - min(workersFree, uint64(len(workers))*workerMargin)
		here := predictedVRAM - min(predictedVRAM, onWorkers) + predictedVRAM/10
		if groupFree > here {
			margin = max(margin, groupFree-here)
		}
	}

	selected := append(withFitMargin(group, margin), workers...)
	slog.Info("splitting model across this machine and cluster workers", "model", model,
		"predicted", format.HumanBytes2(predictedVRAM),
		"local_available", format.HumanBytes2(localAvailable),
		"local_reclaimable", format.HumanBytes2(reclaimable),
		"keep_free_here", format.HumanBytes2(margin),
		"workers_available", format.HumanBytes2(workersFree),
		"placement", place)
	logSelectedGPUGroup(append(local, remote...), selected)
	return selected, launchOpts
}

// tensorSlack is the share of the model, on top of its half, that each
// machine must have free for tensor parallelism; llama-server cannot fit the
// context to the memory in this mode.
const tensorSlack = 10

// selectTensorPlacement places a model with tensor parallelism on one local
// GPU and one worker GPU, each holding half of it. llama.cpp pairs exactly two
// RPC servers, so the local GPU is served by an RPC server of its own on the
// direct link, next to the worker. It returns why that is not possible.
func selectTensorPlacement(cfg *cluster.Config, group, remote []ml.DeviceInfo, predictedVRAM, localCapacity uint64, opts api.Options) ([]ml.DeviceInfo, api.Options, string) {
	if len(group) != 1 {
		return nil, opts, fmt.Sprintf("needs one GPU on this machine, found %d", len(group))
	}
	if len(remote) != 1 {
		return nil, opts, fmt.Sprintf("needs one worker GPU, found %d", len(remote))
	}
	worker := remote[0]
	if worker.FreeMemory == 0 {
		return nil, opts, "the cluster worker has no free RPC slot, all of them hold a model"
	}
	if w, ok := cfg.WorkerByEndpoint(cluster.Endpoints(remote)[0]); !ok || w.LocalAddr == "" {
		return nil, opts, "the worker has no address of this machine on the direct link (local_addr); run 'ollama cluster add' again"
	}
	if predictedVRAM == 0 {
		return nil, opts, "the size of the model is unknown"
	}
	half := predictedVRAM/2 + predictedVRAM*tensorSlack/100
	if half > localCapacity*clusterLocalFitPercent/100 {
		return nil, opts, fmt.Sprintf("half of the model (%s) does not fit on this machine (%s available)", format.HumanBytes2(half), format.HumanBytes2(localCapacity))
	}
	if half+workerMargin > worker.FreeMemory {
		return nil, opts, fmt.Sprintf("half of the model (%s) does not fit on the worker (%s available)", format.HumanBytes2(half), format.HumanBytes2(worker.FreeMemory))
	}

	launchOpts := opts
	launchOpts.MainGPU = nil
	on := true
	launchOpts.ClusterTensor = &on
	return []ml.DeviceInfo{group[0], worker}, launchOpts, ""
}

// withFitMargin returns copies of the devices where the local ones together
// keep margin bytes free, shared in proportion to their free memory.
func withFitMargin(gpus []ml.DeviceInfo, margin uint64) []ml.DeviceInfo {
	gpus = slices.Clone(gpus)
	if margin == 0 {
		return gpus
	}
	var n int
	var free uint64
	for _, g := range gpus {
		if !cluster.IsRemote(g) {
			n++
			free += g.FreeMemory
		}
	}
	for i := range gpus {
		if cluster.IsRemote(gpus[i]) {
			continue
		}
		if free > 0 {
			gpus[i].ClusterFitMargin = uint64(float64(margin) * float64(gpus[i].FreeMemory) / float64(free))
		} else {
			gpus[i].ClusterFitMargin = margin / uint64(n)
		}
	}
	return gpus
}

// bestLocalGroup is the local GPU group the scheduler would pick on its own.
func bestLocalGroup(systemInfo ml.SystemInfo, local []ml.DeviceInfo) []ml.DeviceInfo {
	groups := ml.ByLibrary(local)
	if len(groups) == 0 {
		return nil
	}
	return bestGPUGroupByAvailableMemory(systemInfo, groups)
}

// reclaimableLocked is the memory loaded runners hold on local devices, which
// the scheduler can free by unloading them. s.loadedMu must be held.
func (s *Scheduler) reclaimableLocked(gpus []ml.DeviceInfo) uint64 {
	local, remote := cluster.SplitDevices(gpus)
	if len(remote) == 0 {
		return 0
	}

	var total uint64
	for _, r := range s.loaded {
		r.refMu.Lock()
		if r.llama != nil {
			for _, g := range local {
				total += r.llama.VRAMByGPU(g.DeviceID)
			}
		}
		r.refMu.Unlock()
	}
	return total
}

// clusterFitError explains why a model that needs a cluster worker cannot
// load, instead of letting llama-server spill the rest of a too large model
// into the same unified memory and fail much later. It only applies when a
// cluster is configured, nothing else is loaded (so evicting cannot help) and
// the local GPU shares system memory.
func clusterFitError(modelName string, systemInfo ml.SystemInfo, allGpus, loadGpus []ml.DeviceInfo, predictedVRAM uint64, loadedCount int, opts api.Options) error {
	if loadedCount > 0 || predictedVRAM == 0 || opts.NumGPU == 0 || opts.MainGPU != nil {
		return nil
	}
	if opts.ClusterSplit != nil && !*opts.ClusterSplit {
		return nil
	}
	if _, remote := cluster.SplitDevices(loadGpus); len(remote) > 0 {
		return nil
	}

	cfg, err := cluster.LoadConfig()
	if err != nil || len(cfg.ActiveWorkers()) == 0 {
		return nil
	}

	local, remote := cluster.SplitDevices(allGpus)
	group := bestLocalGroup(systemInfo, local)
	integrated := false
	for _, g := range group {
		integrated = integrated || g.Integrated
	}
	if !integrated {
		return nil
	}
	available, _, _ := availableMemoryForLoad(systemInfo, group)
	if predictedVRAM <= available {
		return nil
	}

	var reasons []string
	if len(remote) > 0 {
		reasons = append(reasons, "the cluster worker has no free RPC slot, all of them hold a model")
	} else {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		for _, c := range cluster.CheckWorkers(ctx, cfg) {
			if !c.Usable {
				reasons = append(reasons, c.Problem+" Run: "+c.Fix)
			}
		}
	}
	if len(reasons) == 0 {
		reasons = append(reasons, "no cluster worker is usable")
	}

	return fmt.Errorf("%s needs about %s of memory, this machine has %s available and %s",
		modelName, format.HumanBytes2(predictedVRAM), format.HumanBytes2(available), strings.Join(reasons, "; "))
}

// workerRetryDelay is how often the workers of a model that fell back to this
// machine are checked; it doubles with each fallback of the model in a row,
// up to workerRetryMax, so a worker that keeps failing at the start does not
// make a large model reload every half minute.
var (
	workerRetryDelay = 30 * time.Second
	workerRetryMax   = 30 * time.Minute
)

// workersReady reports whether the workers with these endpoints answer, run
// this build and have an RPC server free for one more model.
var workersReady = func(ctx context.Context, endpoints []string) bool {
	cfg, err := cluster.LoadConfig()
	if err != nil {
		return false
	}
	for _, ep := range endpoints {
		w, ok := cfg.WorkerByEndpoint(ep)
		if !ok {
			return false
		}
		if c := cluster.CheckWorker(ctx, w); !c.Usable || c.Busy {
			return false
		}
	}
	return true
}

// watchWorkerFallback handles a model the scheduler placed on cluster workers
// that loaded on this machine only, because no worker took it at the start.
// Until the model unloads, its workers are checked; once they can take it,
// the runner asks for a reload, so the next request places the model again
// instead of leaving it here until it expires. runner.refMu must be held.
func (s *Scheduler) watchWorkerFallback(runner *runnerRef, gpus []ml.DeviceInfo) {
	endpoints := cluster.Endpoints(gpus)
	if len(endpoints) == 0 {
		return
	}

	s.workerFallbacksMu.Lock()
	if len(llm.WorkerSlots(runner.pid)) > 0 {
		delete(s.workerFallbacks, runner.modelKey)
		s.workerFallbacksMu.Unlock()
		return
	}
	if s.workerFallbacks == nil {
		s.workerFallbacks = map[string]int{}
	}
	n := s.workerFallbacks[runner.modelKey]
	s.workerFallbacks[runner.modelKey] = n + 1
	s.workerFallbacksMu.Unlock()

	delay := workerRetryMax
	if n < 6 {
		delay = min(workerRetryDelay<<n, workerRetryMax)
	}
	ctx, cancel := context.WithCancel(context.Background())
	runner.fellBack = true
	runner.stopWatch = cancel
	model := runner.modelKey
	if m := runner.model.Load(); m != nil {
		model = m.ShortName
	}
	slog.Warn("model placed on cluster workers runs on this machine only; checking the workers to place it again",
		"model", model, "workers", strings.Join(endpoints, ","), "every", delay)

	go func() {
		t := time.NewTicker(delay)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
			}
			check, done := context.WithTimeout(ctx, 3*time.Second)
			ready := workersReady(check, endpoints)
			done()
			if ready && ctx.Err() == nil {
				runner.workerBack.Store(true)
				slog.Info("cluster workers available again, the model reloads on its next request", "model", model, "workers", strings.Join(endpoints, ","))
				return
			}
		}
	}()
}
