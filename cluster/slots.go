package cluster

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"time"
)

// A worker runs up to MaxSlots RPC servers, one model each: the RPC server
// of llama.cpp serves one client at a time. The first one always runs; the
// worker starts the others when a model asks for a slot and none is free, and
// stops them after they stayed unused for the slot idle time. A slot is
// reserved for the model that got it until its llama-server connects.

// SlotLeaseTTL is how long a slot stays reserved for a llama-server that has
// not connected to it yet.
const SlotLeaseTTL = 30 * time.Second

// SlotStartTimeout is how long the worker waits for a new RPC server to
// listen.
const SlotStartTimeout = 30 * time.Second

// SlotLease is the answer to POST /v1/slots/acquire: the RPC server reserved
// for one model.
type SlotLease struct {
	Index    int    `json:"index"`
	Endpoint string `json:"endpoint"`
}

// SlotRelease is the body of POST /v1/slots/release.
type SlotRelease struct {
	Endpoint string `json:"endpoint"`
}

var slotClient = &http.Client{
	Timeout: SlotStartTimeout + 10*time.Second,
	Transport: &http.Transport{
		Proxy:             nil,
		DisableKeepAlives: true,
	},
}

// AcquireSlots reserves an RPC server on the worker behind each of the
// endpoints (the first RPC server of each worker) and returns the endpoint of
// the reserved one for each. It returns ErrWorkerBusy when a worker has no
// free slot; the slots reserved so far are released.
func AcquireSlots(ctx context.Context, endpoints []string) (map[string]string, error) {
	if len(endpoints) == 0 {
		return nil, nil
	}
	cfg, err := LoadConfig()
	if err != nil {
		return nil, err
	}

	slots := map[string]string{}
	for _, ep := range endpoints {
		w, ok := cfg.WorkerByEndpoint(ep)
		if !ok {
			ReleaseSlots(ctx, slots)
			return nil, fmt.Errorf("worker %s is no longer configured", ep)
		}
		lease, err := acquireSlot(ctx, w)
		if err != nil {
			ReleaseSlots(ctx, slots)
			return nil, err
		}
		slots[ep] = lease.Endpoint
		slog.Info("reserved RPC slot on cluster worker", "worker", w.Label(), "slot", lease.Index, "endpoint", lease.Endpoint)
	}
	return slots, nil
}

func acquireSlot(ctx context.Context, w Worker) (SlotLease, error) {
	var lease SlotLease
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://"+w.StatusAddr()+"/v1/slots/acquire", nil)
	if err != nil {
		return lease, err
	}
	resp, err := slotClient.Do(req)
	if err != nil {
		return lease, fmt.Errorf("worker %s: reserving an RPC slot: %w", w.Label(), err)
	}
	defer resp.Body.Close()
	switch resp.StatusCode {
	case http.StatusOK:
	case http.StatusConflict:
		return lease, fmt.Errorf("worker %s: %w", w.Label(), ErrWorkerBusy)
	default:
		return lease, fmt.Errorf("worker %s: reserving an RPC slot: %s", w.Label(), resp.Status)
	}
	if err := json.NewDecoder(resp.Body).Decode(&lease); err != nil {
		return lease, fmt.Errorf("worker %s: reserving an RPC slot: %w", w.Label(), err)
	}
	if lease.Endpoint == "" {
		return lease, fmt.Errorf("worker %s: reserving an RPC slot: no endpoint in the answer", w.Label())
	}
	return lease, nil
}

// ReleaseSlots gives back slots reserved by AcquireSlots for a llama-server
// that did not start. A slot its llama-server connected to is released when
// the connection closes; this only shortens the reservation.
func ReleaseSlots(ctx context.Context, slots map[string]string) {
	if len(slots) == 0 {
		return
	}
	cfg, err := LoadConfig()
	if err != nil {
		return
	}
	for ep, slot := range slots {
		w, ok := cfg.WorkerByEndpoint(ep)
		if !ok {
			continue
		}
		body, _ := json.Marshal(SlotRelease{Endpoint: slot})
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://"+w.StatusAddr()+"/v1/slots/release", bytes.NewReader(body))
		if err != nil {
			continue
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := httpClient.Do(req)
		if err != nil {
			slog.Debug("releasing RPC slot", "worker", w.Label(), "endpoint", slot, "error", err)
			continue
		}
		resp.Body.Close()
	}
}
