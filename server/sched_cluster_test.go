package server

import (
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cluster"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
)

func TestSelectClusterPlacement(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	sys := ml.SystemInfo{TotalMemory: 128 * format.GigaByte, FreeMemory: 80 * format.GigaByte}
	local := []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}, Name: "CUDA0", Integrated: true, TotalMemory: 120 * format.GigaByte, FreeMemory: 80 * format.GigaByte}}
	remote := []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "10.100.48.2:50052/0", Library: cluster.RemoteLibrary}, Name: "RPC0[10.100.48.2:50052]", TotalMemory: 120 * format.GigaByte, FreeMemory: 110 * format.GigaByte}}
	opts := api.DefaultOptions()

	countRemote := func(gpus []ml.DeviceInfo) int {
		_, r := cluster.SplitDevices(gpus)
		return len(r)
	}

	// fits here: stays local
	gpus, _ := selectClusterPlacement("m", sys, local, remote, 30*format.GigaByte, 0, opts)
	if countRemote(gpus) != 0 {
		t.Fatalf("small model must stay local: %v", gpus)
	}

	// does not fit: split
	gpus, launch := selectClusterPlacement("m", sys, local, remote, 140*format.GigaByte, 0, opts)
	if countRemote(gpus) != 1 || len(gpus) != 2 || launch.MainGPU != nil {
		t.Fatalf("large model must be split: %v", gpus)
	}

	// fits once other models are unloaded: stays local
	gpus, _ = selectClusterPlacement("m", sys, local, remote, 100*format.GigaByte, 40*format.GigaByte, opts)
	if countRemote(gpus) != 0 {
		t.Fatalf("model fitting after eviction must stay local: %v", gpus)
	}

	// forced split and forced local
	yes, no := true, false
	opts.ClusterSplit = &yes
	if gpus, _ = selectClusterPlacement("m", sys, local, remote, 1*format.GigaByte, 0, opts); countRemote(gpus) != 1 {
		t.Fatalf("cluster_split=true must split: %v", gpus)
	}
	opts.ClusterSplit = &no
	if gpus, _ = selectClusterPlacement("m", sys, local, remote, 140*format.GigaByte, 0, opts); countRemote(gpus) != 0 {
		t.Fatalf("cluster_split=false must stay local: %v", gpus)
	}

	// busy worker (no free memory): stays local
	opts.ClusterSplit = nil
	busy := []ml.DeviceInfo{remote[0]}
	busy[0].FreeMemory = 0
	if gpus, _ = selectClusterPlacement("m", sys, local, busy, 140*format.GigaByte, 0, opts); countRemote(gpus) != 0 {
		t.Fatalf("busy worker must not be used: %v", gpus)
	}
}

func TestSelectClusterPlacementPolicy(t *testing.T) {
	cfgPath := filepath.Join(t.TempDir(), "cluster.json")
	t.Setenv("OLLAMA_CLUSTER_CONFIG", cfgPath)
	sys := ml.SystemInfo{TotalMemory: 128 * format.GigaByte, FreeMemory: 100 * format.GigaByte}
	local := []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}, Name: "CUDA0", Integrated: true, TotalMemory: 120 * format.GigaByte, FreeMemory: 100 * format.GigaByte}}
	remote := []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "10.100.48.2:50052/0", Library: cluster.RemoteLibrary}, Name: "RPC0[10.100.48.2:50052]", TotalMemory: 120 * format.GigaByte, FreeMemory: 110 * format.GigaByte}}
	opts := api.DefaultOptions()
	save := func(c cluster.Config) {
		t.Helper()
		if err := c.SaveFile(cfgPath); err != nil {
			t.Fatal(err)
		}
	}
	margin := func(gpus []ml.DeviceInfo) uint64 {
		for _, g := range gpus {
			if !cluster.IsRemote(g) {
				return g.ClusterFitMargin
			}
		}
		return 0
	}
	count := func(gpus []ml.DeviceInfo) (int, int) {
		l, r := cluster.SplitDevices(gpus)
		return len(l), len(r)
	}

	// keep 40 GB free here: a 70 GB model no longer fits and is split with the margin
	save(cluster.Config{KeepFreeHere: 40 * format.GigaByte})
	gpus, _ := selectClusterPlacement("m", sys, local, remote, 70*format.GigaByte, 0, opts)
	if l, r := count(gpus); l != 1 || r != 1 || margin(gpus) != 40*format.GigaByte {
		t.Fatalf("keep free here: %d local, %d remote, margin %d", l, r, margin(gpus))
	}
	gpus, _ = selectClusterPlacement("m", sys, local, remote, 20*format.GigaByte, 0, opts)
	if l, r := count(gpus); l != 1 || r != 0 || margin(gpus) != 40*format.GigaByte {
		t.Fatalf("small model stays here with the margin: %d local, %d remote, margin %d", l, r, margin(gpus))
	}

	// worker-first from 30 GB
	save(cluster.Config{Policy: cluster.PolicyWorkerFirst, WorkerFirstFrom: 30 * format.GigaByte})
	if gpus, _ = selectClusterPlacement("m", sys, local, remote, 20*format.GigaByte, 0, opts); !slices.Equal([]int{1, 0}, []int{len(gpus), countRemoteDevices(gpus)}) {
		t.Fatalf("small model must stay here: %v", gpus)
	}
	gpus, _ = selectClusterPlacement("m", sys, local, remote, 60*format.GigaByte, 0, opts)
	if l, r := count(gpus); l != 0 || r != 1 {
		t.Fatalf("model fitting on the worker must go there only: %d local, %d remote", l, r)
	}
	gpus, _ = selectClusterPlacement("m", sys, local, remote, 150*format.GigaByte, 0, opts)
	// the worker takes 109 GB, here keeps the rest plus a tenth of the model
	want := 100*format.GigaByte - (150*format.GigaByte - (110*format.GigaByte - 1024*format.MebiByte) + 15*format.GigaByte)
	if l, r := count(gpus); l != 1 || r != 1 || margin(gpus) != uint64(want) {
		t.Fatalf("too large for the worker: %d local, %d remote, margin %d want %d", l, r, margin(gpus), want)
	}

	// model rules win; a busy worker sends the model here
	save(cluster.Config{Policy: cluster.PolicyWorkerFirst, Models: map[string]string{"pinned": cluster.PlaceHere, "tiny": cluster.PlaceWorker}})
	if gpus, _ = selectClusterPlacement("pinned:latest", sys, local, remote, 60*format.GigaByte, 0, opts); countRemoteDevices(gpus) != 0 {
		t.Fatalf("rule here: %v", gpus)
	}
	if gpus, _ = selectClusterPlacement("tiny", sys, local, remote, 1*format.GigaByte, 0, opts); len(gpus) != 1 || countRemoteDevices(gpus) != 1 {
		t.Fatalf("rule worker: %v", gpus)
	}
	busy := []ml.DeviceInfo{remote[0]}
	busy[0].FreeMemory = 0
	if gpus, _ = selectClusterPlacement("tiny", sys, local, busy, 1*format.GigaByte, 0, opts); countRemoteDevices(gpus) != 0 {
		t.Fatalf("busy worker: %v", gpus)
	}
}

func countRemoteDevices(gpus []ml.DeviceInfo) int {
	_, r := cluster.SplitDevices(gpus)
	return len(r)
}

func TestWithFitMarginSharesAcrossDevices(t *testing.T) {
	gpus := []ml.DeviceInfo{
		{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}, FreeMemory: 30 * format.GigaByte},
		{DeviceID: ml.DeviceID{ID: "1", Library: "CUDA"}, FreeMemory: 10 * format.GigaByte},
		{DeviceID: ml.DeviceID{ID: "w/0", Library: cluster.RemoteLibrary}, FreeMemory: 100 * format.GigaByte},
	}
	got := withFitMargin(gpus, 8*format.GigaByte)
	if got[0].ClusterFitMargin != 6*format.GigaByte || got[1].ClusterFitMargin != 2*format.GigaByte || got[2].ClusterFitMargin != 0 {
		t.Fatalf("margins %d %d %d", got[0].ClusterFitMargin, got[1].ClusterFitMargin, got[2].ClusterFitMargin)
	}
	if gpus[0].ClusterFitMargin != 0 {
		t.Fatal("input devices changed")
	}
}

func TestWorkerRuleUnknownSize(t *testing.T) {
	cfgPath := filepath.Join(t.TempDir(), "cluster.json")
	t.Setenv("OLLAMA_CLUSTER_CONFIG", cfgPath)
	if err := (&cluster.Config{Models: map[string]string{"m": cluster.PlaceWorker}}).SaveFile(cfgPath); err != nil {
		t.Fatal(err)
	}
	sys := ml.SystemInfo{TotalMemory: 128 * format.GigaByte, FreeMemory: 100 * format.GigaByte}
	local := []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}, Name: "CUDA0", TotalMemory: 120 * format.GigaByte, FreeMemory: 100 * format.GigaByte}}
	remote := []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "10.100.48.2:50052/0", Library: cluster.RemoteLibrary}, TotalMemory: 120 * format.GigaByte, FreeMemory: 110 * format.GigaByte}}
	gpus, _ := selectClusterPlacement("m", sys, local, remote, 0, 0, api.DefaultOptions())
	if !workersOnly(gpus) {
		t.Fatalf("unknown size with a worker rule must go to the worker: %v", gpus)
	}
}
