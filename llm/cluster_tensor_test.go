package llm

import (
	"slices"
	"testing"
)

func TestTensorDeviceArgs(t *testing.T) {
	got := tensorDeviceArgs("10.100.48.2:50052", "10.100.48.1:41000")
	want := []string{"--rpc", "10.100.48.2:50052,10.100.48.1:41000", "--device", "RPC0,RPC1", "--split-mode", "tensor", "--tensor-split", "1,1", "-ngl", "999", "--fit", "off"}
	if !slices.Equal(got, want) {
		t.Fatalf("got %v", got)
	}
}

func TestTensorShare(t *testing.T) {
	registerTensorPeer(4242, &tensorPeer{endpoint: "10.100.48.1:41000"})
	defer func() {
		tensorPeersMu.Lock()
		delete(tensorPeers, 4242)
		tensorPeersMu.Unlock()
	}()
	byDevice := map[string]uint64{tensorBuffer: 3000, "CUDA0": 100}
	if got := tensorShare(4242, byDevice); got != 3000 {
		t.Errorf("tensor runner: %d", got)
	}
	if got := tensorShare(1, byDevice); got != 0 {
		t.Errorf("runner without tensor parallelism: %d", got)
	}
	if !TensorParallel(4242) || TensorParallel(1) {
		t.Error("TensorParallel")
	}
}
