package cluster

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/ml"
)

func TestPlacement(t *testing.T) {
	const gb = 1_000_000_000
	c := &Config{}
	if p := c.Placement("m", 100*gb); p != PlaceAuto {
		t.Fatalf("default: %s", p)
	}

	c.Policy = PolicyWorkerFirst
	c.WorkerFirstFrom = 30 * gb
	if p := c.Placement("small", 10*gb); p != PlaceAuto {
		t.Fatalf("small model under worker-first: %s", p)
	}
	if p := c.Placement("big", 30*gb); p != PlaceWorker {
		t.Fatalf("big model under worker-first: %s", p)
	}

	c.Models = map[string]string{"big": PlaceHere, "small:latest": PlaceSplit, "pinned:7b": PlaceAuto}
	if p := c.Placement("big:latest", 80*gb); p != PlaceHere {
		t.Fatalf("rule for big: %s", p)
	}
	if p := c.Placement("small", 1*gb); p != PlaceSplit {
		t.Fatalf("rule for small: %s", p)
	}
	if p := c.Placement("pinned:7b", 40*gb); p != PlaceWorker {
		t.Fatalf("auto rule falls back to the policy: %s", p)
	}

	c.Policy = PolicySpread
	if p := c.Placement("other", 1*gb); p != PlaceSplit {
		t.Fatalf("spread: %s", p)
	}
}

func TestValidate(t *testing.T) {
	for _, c := range []Config{{}, {Policy: PolicyWorkerFirst, Models: map[string]string{"m": PlaceWorker}}} {
		if err := c.Validate(); err != nil {
			t.Fatalf("%+v: %v", c, err)
		}
	}
	for _, c := range []Config{{Policy: "fast"}, {Models: map[string]string{"m": "there"}}} {
		if err := c.Validate(); err == nil {
			t.Fatalf("%+v must be rejected", c)
		}
	}
}

func TestValidateWorkers(t *testing.T) {
	good := Worker{Name: "gx10-bf3f", SSH: "jarek@10.100.48.2", Addr: "10.100.48.2", LocalAddr: "10.100.48.1", RPCPort: 50052, StatusPort: 50053, Mode: "docker", Image: "ollama-cluster:0.40.3-cluster"}
	c := Config{Workers: []Worker{good}, Locations: []Location{{Path: "/mnt/dgx-data/models"}}}
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	bad := []func(w *Worker){
		func(w *Worker) { w.LocalAddr = "10.100.48.1 -j ACCEPT; touch /tmp/x" },
		func(w *Worker) { w.Addr = "$(id)" },
		func(w *Worker) { w.SSH = "-oProxyCommand=sh" },
		func(w *Worker) { w.SSH = "jarek@host;id" },
		func(w *Worker) { w.Name = "a b" },
		func(w *Worker) { w.Image = "img;rm -rf /" },
		func(w *Worker) { w.RPCPort = 70000 },
		func(w *Worker) { w.Mode = "ssh" },
	}
	for i, f := range bad {
		w := good
		f(&w)
		if err := (&Config{Workers: []Worker{w}}).Validate(); err == nil {
			t.Fatalf("case %d must be rejected: %+v", i, w)
		}
	}
	for _, p := range []string{"models", "/mnt/../etc", "/mnt/x/", "/etc/ollama", "/", "/home", "/mnt/a:b", "/var/run/docker.sock"} {
		if err := (&Config{Locations: []Location{{Path: p}}}).Validate(); err == nil {
			t.Fatalf("location %q must be rejected", p)
		}
	}
}

func TestLlamaDeviceNames(t *testing.T) {
	dev := func(id string) ml.DeviceInfo {
		return ml.DeviceInfo{DeviceID: ml.DeviceID{ID: id, Library: RemoteLibrary}}
	}
	gpus := []ml.DeviceInfo{
		{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}},
		dev("10.100.48.2:50052/0"), dev("10.100.48.2:50052/1"), dev("10.100.48.3:50052/0"),
	}
	got := LlamaDeviceNames(gpus)
	if strings.Join(got, ",") != "RPC0,RPC1,RPC2" {
		t.Fatalf("names: %v", got)
	}
	// second device of the first worker not selected: numbering still counts it
	if got := LlamaDeviceNames([]ml.DeviceInfo{gpus[2], gpus[3]}); strings.Join(got, ",") != "RPC1,RPC2" {
		t.Fatalf("names: %v", got)
	}
}

func TestPlacementTensor(t *testing.T) {
	c := &Config{Tensor: true, Policy: PolicyWorkerFirst, WorkerFirstFrom: 50, Models: map[string]string{"a": PlaceSplit}}
	if p := c.Placement("a", 10); p != PlaceSplit {
		t.Errorf("rule must win over tensor: %s", p)
	}
	if p := c.Placement("b", 10); p != PlaceTensor {
		t.Errorf("tensor on: %s", p)
	}
	if p := c.PolicyPlacement(100); p != PlaceWorker {
		t.Errorf("policy placement: %s", p)
	}
	c.Models["a"] = PlaceTensor
	if err := c.Validate(); err != nil {
		t.Errorf("rule tensor must be valid: %v", err)
	}
}
