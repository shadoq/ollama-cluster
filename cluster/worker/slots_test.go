package worker

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/cluster"
)

// fakeSockets is the socket state of the RPC ports in a test.
type fakeSockets struct {
	mu        sync.Mutex
	listening map[int]bool
	clients   map[int]int
}

func (f *fakeSockets) get(port int) (bool, int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.listening[port], f.clients[port]
}

func (f *fakeSockets) set(port int, listening bool, clients int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.listening[port] = listening
	f.clients[port] = clients
}

// testWorker returns a worker whose RPC servers listen as soon as they run
// and stop listening when they end, with a clock the test moves.
func testWorker(t *testing.T, maxSlots int) (*Worker, *fakeSockets, *time.Time) {
	t.Helper()
	w := newWorker(Options{Addr: "10.0.0.2", MaxSlots: maxSlots, SlotIdle: time.Minute})
	sockets := &fakeSockets{listening: map[int]bool{}, clients: map[int]int{}}
	sockets.set(cluster.DefaultRPCPort, true, 0)
	now := time.Date(2026, 10, 25, 12, 0, 0, 0, time.UTC)
	w.now = func() time.Time { return now }
	w.sockets = sockets.get
	w.run = func(ctx context.Context, s *slot) error {
		sockets.set(s.port, true, 0)
		<-ctx.Done()
		sockets.set(s.port, false, 0)
		return ctx.Err()
	}
	return w, sockets, &now
}

func slotCount(w *Worker) int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return len(w.slots)
}

func TestAcquireUsesFreeSlotFirst(t *testing.T) {
	w, _, _ := testWorker(t, 3)
	lease, err := w.acquire(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if lease.Index != 0 || lease.Endpoint != "10.0.0.2:50052" {
		t.Fatalf("got %+v, want the first slot", lease)
	}
	if n := slotCount(w); n != 1 {
		t.Fatalf("started %d slots, want only the first one", n)
	}
}

func TestAcquireStartsSlotsUpToLimit(t *testing.T) {
	w, sockets, _ := testWorker(t, 3)
	sockets.set(cluster.DefaultRPCPort, true, 1) // a model on the first slot

	want := []string{"10.0.0.2:50054", "10.0.0.2:50055"} // 50053 is the status port
	for i, ep := range want {
		lease, err := w.acquire(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		if lease.Index != i+1 || lease.Endpoint != ep {
			t.Fatalf("got %+v, want slot %d on %s", lease, i+1, ep)
		}
	}
	// both new slots stay reserved until their llama-server connects
	if _, err := w.acquire(t.Context()); !errors.Is(err, errNoSlot) {
		t.Fatalf("got %v, want errNoSlot", err)
	}
	st := w.snapshot()
	if st.MaxSlots != 3 || st.FreeSlots != 0 || len(st.Slots) != 3 {
		t.Fatalf("status %d of %d free, %d slots", st.FreeSlots, st.MaxSlots, len(st.Slots))
	}
}

func TestLeaseEndsWhenClientConnectsOrTimeRunsOut(t *testing.T) {
	w, sockets, now := testWorker(t, 1)
	if _, err := w.acquire(t.Context()); err != nil {
		t.Fatal(err)
	}
	if _, err := w.acquire(t.Context()); !errors.Is(err, errNoSlot) {
		t.Fatalf("reserved slot given twice: %v", err)
	}

	// the reservation ran out without a client
	*now = now.Add(cluster.SlotLeaseTTL + time.Second)
	if _, err := w.acquire(t.Context()); err != nil {
		t.Fatalf("slot not free after the reservation ran out: %v", err)
	}

	// the client connected and left again
	sockets.set(cluster.DefaultRPCPort, true, 1)
	if st := w.snapshot(); st.FreeSlots != 0 || st.Slots[0].Reserved {
		t.Fatalf("connected slot: %+v", st.Slots[0])
	}
	sockets.set(cluster.DefaultRPCPort, true, 0)
	if st := w.snapshot(); st.FreeSlots != 1 {
		t.Fatalf("slot not free after its client left: %+v", st.Slots[0])
	}
}

func TestReleaseEndsReservation(t *testing.T) {
	w, _, _ := testWorker(t, 1)
	lease, err := w.acquire(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	w.release(lease.Endpoint)
	if _, err := w.acquire(t.Context()); err != nil {
		t.Fatalf("released slot not free: %v", err)
	}
}

func TestReapStopsIdleExtraSlots(t *testing.T) {
	w, sockets, now := testWorker(t, 2)
	sockets.set(cluster.DefaultRPCPort, true, 1)
	lease, err := w.acquire(t.Context())
	if err != nil || lease.Index != 1 {
		t.Fatalf("got %+v, %v", lease, err)
	}
	port := 50054
	sockets.set(port, true, 1) // its model runs

	*now = now.Add(time.Hour)
	w.reap()
	if n := slotCount(w); n != 2 {
		t.Fatalf("slot with a client stopped")
	}

	sockets.set(port, true, 0) // the model unloaded
	sockets.set(cluster.DefaultRPCPort, true, 0)
	w.reap()
	*now = now.Add(59 * time.Second)
	w.reap()
	if n := slotCount(w); n != 2 {
		t.Fatalf("slot stopped before the idle time")
	}
	*now = now.Add(time.Second)
	w.reap()
	deadline := time.Now().Add(5 * time.Second)
	for slotCount(w) != 1 {
		if time.Now().After(deadline) {
			t.Fatal("idle slot not stopped")
		}
		time.Sleep(10 * time.Millisecond)
	}

	// the first slot always runs, however long it is idle
	*now = now.Add(24 * time.Hour)
	w.reap()
	w.reap()
	if n := slotCount(w); n != 1 {
		t.Fatalf("first slot stopped")
	}
}

func TestAcquireReportsSlotThatDoesNotStart(t *testing.T) {
	w, sockets, _ := testWorker(t, 2)
	sockets.set(cluster.DefaultRPCPort, true, 1)
	w.run = func(ctx context.Context, s *slot) error { return errors.New("no GPU") }
	if _, err := w.acquire(t.Context()); err == nil || errors.Is(err, errNoSlot) {
		t.Fatalf("got %v, want the start error", err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for slotCount(w) != 1 {
		if time.Now().After(deadline) {
			t.Fatal("failed slot kept")
		}
		time.Sleep(10 * time.Millisecond)
	}
}
