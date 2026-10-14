package llm

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/cluster"
	"github.com/ollama/ollama/ml"
)

func TestAppendClusterFitArgs(t *testing.T) {
	local := ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}, Name: "CUDA0"}
	remote := ml.DeviceInfo{DeviceID: ml.DeviceID{ID: "10.100.48.2:50052/0", Library: cluster.RemoteLibrary}, Name: "RPC0[10.100.48.2:50052]"}

	// no margin: llama-server defaults
	launch := llamaServerLaunchConfig{gpus: []ml.DeviceInfo{local, remote}}
	if got := appendClusterFitArgs([]string{"--rpc", "x"}, launch); len(got) != 2 {
		t.Fatalf("no margin: %v", got)
	}

	// split: worker devices come first in llama-server
	local.ClusterFitMargin = 40 << 30
	launch.gpus = []ml.DeviceInfo{local, remote}
	got := appendClusterFitArgs([]string{"--rpc", "x"}, launch)
	if !slices.Equal(got[2:], []string{"--fit-target", "1024,40960"}) {
		t.Fatalf("split: %v", got)
	}

	// the worker dropped at start: only this machine
	got = appendClusterFitArgs(nil, launch)
	if !slices.Equal(got, []string{"--fit-target", "40960"}) {
		t.Fatalf("worker dropped: %v", got)
	}

	// a margin below the default keeps the default
	local.ClusterFitMargin = 100 << 20
	launch.gpus = []ml.DeviceInfo{local}
	if got = appendClusterFitArgs(nil, launch); !slices.Equal(got, []string{"--fit-target", "1024"}) {
		t.Fatalf("small margin: %v", got)
	}
}
