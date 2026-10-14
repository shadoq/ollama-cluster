package cluster

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/version"
)

// fakeWorker serves a worker status that reports a client for the first
// busyFor requests.
func fakeWorker(t *testing.T, busyFor int32) (Worker, *atomic.Int32) {
	t.Helper()
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		clients := 0
		if calls.Add(1) <= busyFor {
			clients = 1
		}
		json.NewEncoder(w).Encode(WorkerStatus{
			Version:  version.Version,
			LlamaCpp: LlamaCppVersion,
			Running:  true,
			Clients:  clients,
			Devices:  []RemoteDevice{{Name: "CUDA0", TotalMemory: 100, FreeMemory: 90}},
		})
	}))
	t.Cleanup(srv.Close)
	host, port, _ := net.SplitHostPort(srv.Listener.Addr().String())
	p, _ := strconv.Atoi(port)
	return Worker{Name: "w", Addr: host, StatusPort: p, RPCPort: p + 1}, &calls
}

func TestCheckAfterRelease(t *testing.T) {
	t.Run("busy without a release", func(t *testing.T) {
		w, calls := fakeWorker(t, 100)
		c := checkAfterRelease(context.Background(), w)
		if !c.Busy || calls.Load() != 1 {
			t.Fatalf("busy=%v calls=%d, want busy after one probe", c.Busy, calls.Load())
		}
	})

	t.Run("busy right after a release", func(t *testing.T) {
		w, calls := fakeWorker(t, 2)
		Released([]string{w.Endpoint()})
		c := checkAfterRelease(context.Background(), w)
		if c.Busy || calls.Load() != 3 {
			t.Fatalf("busy=%v calls=%d, want free on the third probe", c.Busy, calls.Load())
		}
		releasedMu.Lock()
		_, left := released[w.Endpoint()]
		releasedMu.Unlock()
		if left {
			t.Fatal("release still recorded after the worker was found free")
		}
	})

	t.Run("still busy after the grace", func(t *testing.T) {
		w, _ := fakeWorker(t, 1000)
		releasedMu.Lock()
		released[w.Endpoint()] = time.Now().Add(-releaseGrace)
		releasedMu.Unlock()
		if c := checkAfterRelease(context.Background(), w); !c.Busy {
			t.Fatal("worker reported free")
		}
		releasedMu.Lock()
		_, left := released[w.Endpoint()]
		releasedMu.Unlock()
		if left {
			t.Fatal("expired release still recorded")
		}
	})
}
