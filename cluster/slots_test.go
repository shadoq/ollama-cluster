package cluster

import (
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"slices"
	"strconv"
	"testing"

	"github.com/ollama/ollama/ml"
)

func TestSlotPorts(t *testing.T) {
	if got := SlotPorts(50052, 50053, 3); !slices.Equal(got, []int{50052, 50054, 50055}) {
		t.Fatalf("default ports: got %v", got)
	}
	if got := SlotPorts(6000, 7000, 2); !slices.Equal(got, []int{6000, 6001}) {
		t.Fatalf("status port elsewhere: got %v", got)
	}
	w := Worker{Addr: "10.0.0.2"}
	if got := w.SlotPorts(); !slices.Equal(got, []int{50052, 50054, 50055}) {
		t.Fatalf("worker defaults: got %v", got)
	}
}

func TestBusyBySlots(t *testing.T) {
	cases := []struct {
		st   WorkerStatus
		busy bool
	}{
		{WorkerStatus{Clients: 1}, true},                             // no slots: one RPC server
		{WorkerStatus{Clients: 0}, false},                            //
		{WorkerStatus{Clients: 1, MaxSlots: 3, FreeSlots: 2}, false}, // a model on the first slot
		{WorkerStatus{Clients: 1, MaxSlots: 3, FreeSlots: 0}, true},
	}
	for i, c := range cases {
		if got := c.st.busy(); got != c.busy {
			t.Fatalf("case %d: busy=%v, want %v", i, got, c.busy)
		}
	}
	st := WorkerStatus{MaxSlots: 3, FreeSlots: 1}
	if got := st.SlotsLabel(); got != "2 of 3 slots used" {
		t.Fatalf("label %q", got)
	}
}

func TestSlotDevices(t *testing.T) {
	gpus := []ml.DeviceInfo{
		{DeviceID: ml.DeviceID{ID: "0", Library: "CUDA"}, Name: "CUDA0"},
		{DeviceID: ml.DeviceID{ID: "10.100.48.2:50052/0", Library: RemoteLibrary}, Name: "RPC0[10.100.48.2:50052]"},
	}
	got := SlotDevices(gpus, map[string]string{"10.100.48.2:50052": "10.100.48.2:50054"})
	if got[0].Name != "CUDA0" || got[1].Name != "RPC0[10.100.48.2:50054]" || got[1].ID != "10.100.48.2:50052/0" {
		t.Fatalf("got %+v", got)
	}
	if gpus[1].Name != "RPC0[10.100.48.2:50052]" {
		t.Fatal("input changed")
	}
	// a restart on the first slot gets the first name back
	if again := SlotDevices(got, nil); again[1].Name != "RPC0[10.100.48.2:50052]" {
		t.Fatalf("got %+v", again)
	}
}

// slotWorker serves the slot endpoints of a worker and returns a config that
// points at it.
func slotWorker(t *testing.T, handler http.HandlerFunc) Worker {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	host, port, _ := net.SplitHostPort(srv.Listener.Addr().String())
	p, _ := strconv.Atoi(port)
	w := Worker{Name: "w1", Addr: host, RPCPort: 50052, StatusPort: p}

	path := filepath.Join(t.TempDir(), "cluster.json")
	if err := (&Config{Workers: []Worker{w}}).SaveFile(path); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OLLAMA_CLUSTER_CONFIG", path)
	return w
}

func TestAcquireSlots(t *testing.T) {
	var released []string
	w := slotWorker(t, func(rw http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/slots/acquire":
			_ = json.NewEncoder(rw).Encode(SlotLease{Index: 1, Endpoint: "127.0.0.1:50054"})
		case "/v1/slots/release":
			var req SlotRelease
			_ = json.NewDecoder(r.Body).Decode(&req)
			released = append(released, req.Endpoint)
			rw.WriteHeader(http.StatusNoContent)
		}
	})

	slots, err := AcquireSlots(t.Context(), []string{w.Endpoint()})
	if err != nil {
		t.Fatal(err)
	}
	if slots[w.Endpoint()] != "127.0.0.1:50054" {
		t.Fatalf("got %v", slots)
	}
	ReleaseSlots(t.Context(), slots)
	if !slices.Equal(released, []string{"127.0.0.1:50054"}) {
		t.Fatalf("released %v", released)
	}
}

func TestAcquireSlotsRejectsForeignEndpoint(t *testing.T) {
	for _, ep := range []string{
		"10.0.0.9:50054",                      // another host
		"127.0.0.1:50054,10.0.0.9:50052",      // a second --rpc endpoint
		"127.0.0.1:22",                        // not a slot port
		"127.0.0.1:050054",                    //
		"127.0.0.1:50054 --model /etc/shadow", // more arguments
	} {
		w := slotWorker(t, func(rw http.ResponseWriter, r *http.Request) {
			_ = json.NewEncoder(rw).Encode(SlotLease{Index: 1, Endpoint: ep})
		})
		if slots, err := AcquireSlots(t.Context(), []string{w.Endpoint()}); err == nil {
			t.Fatalf("endpoint %q accepted: %v", ep, slots)
		}
	}
}

func TestAcquireSlotsBusy(t *testing.T) {
	w := slotWorker(t, func(rw http.ResponseWriter, r *http.Request) {
		http.Error(rw, "all RPC servers of the worker serve a model", http.StatusConflict)
	})
	if _, err := AcquireSlots(t.Context(), []string{w.Endpoint()}); !errors.Is(err, ErrWorkerBusy) {
		t.Fatalf("got %v, want ErrWorkerBusy", err)
	}
}
