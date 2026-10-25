package clustercmd

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/cluster"
)

func TestFirewallRuleCoversSlots(t *testing.T) {
	w := cluster.Worker{Addr: "10.100.48.2", LocalAddr: "10.100.48.1", RPCPort: 50052, StatusPort: 50053}
	want := "INPUT ! -i lo -p tcp -m multiport --dports 50052,50054,50055,51052,51054,51055,50053 ! -s 10.100.48.1 -j DROP"
	if got := firewallRule(w); got != want {
		t.Fatalf("got  %s\nwant %s", got, want)
	}
	if got := legacyFirewallRule(w); got != "INPUT ! -i lo -p tcp -m multiport --dports 50052,50053 ! -s 10.100.48.1 -j DROP" {
		t.Fatalf("legacy rule %s", got)
	}

	// iptables multiport takes 15 ports
	w.MaxSlots = cluster.MaxSlotLimit
	if err := w.Validate(); err != nil {
		t.Fatal(err)
	}
	w.MaxSlots = cluster.MaxSlotLimit + 1
	if err := w.Validate(); err == nil {
		t.Fatal("max_slots above the limit accepted")
	}
}

func TestWorkerArgsSlots(t *testing.T) {
	w := cluster.Worker{Addr: "10.100.48.2", RPCPort: 50052, StatusPort: 50053}
	if args := workerArgs(w, false); slices.Contains(args, "--max-slots") || slices.Contains(args, "--slot-idle") {
		t.Fatalf("defaults passed: %v", args)
	}
	w.MaxSlots, w.SlotIdle = 2, 600
	args := workerArgs(w, false)
	if i := slices.Index(args, "--max-slots"); i < 0 || args[i+1] != "2" {
		t.Fatalf("max slots: %v", args)
	}
	if i := slices.Index(args, "--slot-idle"); i < 0 || args[i+1] != "10m0s" {
		t.Fatalf("slot idle: %v", args)
	}
}

func TestPlacementNamesSlot(t *testing.T) {
	w := cluster.Worker{Name: "gx10-bf3f", Addr: "10.100.48.2"}
	st := &cluster.StatusResponse{Workers: []cluster.WorkerCheck{{Worker: w}}}
	l := cluster.LoadedModel{
		Tensor: true,
		Devices: []cluster.DeviceMemory{
			{Name: "CUDA0", Size: 9_500_000_000},
			{Name: "RPC0[10.100.48.2:50052]", Remote: true, Size: 9_700_000_000},
		},
		WorkerSlots: []string{"10.100.48.2:50054"},
	}
	if got := placement(st, l); got != "tensor: here 9.5 GB + gx10-bf3f#1 9.7 GB" {
		t.Fatalf("placement: %s", got)
	}
	l.WorkerSlots = []string{"10.100.48.2:50052"}
	if got := placement(st, l); got != "tensor: here 9.5 GB + gx10-bf3f 9.7 GB" {
		t.Fatalf("first slot: %s", got)
	}
}

func TestBusyState(t *testing.T) {
	cases := []struct {
		c    cluster.WorkerCheck
		want string
	}{
		{cluster.WorkerCheck{Status: &cluster.WorkerStatus{MaxSlots: 3, FreeSlots: 3}}, "idle, 3 slots"},
		{cluster.WorkerCheck{Status: &cluster.WorkerStatus{MaxSlots: 3, FreeSlots: 2}}, "1 of 3 slots used"},
		{cluster.WorkerCheck{Status: &cluster.WorkerStatus{MaxSlots: 3}, Busy: true}, "3 of 3 slots used"},
		{cluster.WorkerCheck{Status: &cluster.WorkerStatus{}, Busy: true}, "holds a model"},
	}
	for _, c := range cases {
		if got := busyState(c.c); got != c.want {
			t.Errorf("got %q, want %q", got, c.want)
		}
	}
}
