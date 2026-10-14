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
	if place == cluster.PlaceHere {
		return localOnly("placement rule")
	}

	group := bestLocalGroup(systemInfo, local)
	localAvailable, _, _ := availableMemoryForLoad(systemInfo, group)
	capacity := localAvailable + reclaimable
	capacity -= min(keepFree, capacity)
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
		reasons = append(reasons, "the cluster worker is busy with another model")
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
