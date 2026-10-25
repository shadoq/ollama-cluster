package cluster

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"slices"
	"strconv"
	"sync"
	"time"

	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/version"
)

// LlamaCppVersion is the llama.cpp release this binary was built against
// (the LLAMA_CPP_VERSION file). The main package sets it at startup.
var LlamaCppVersion string

// RemoteLibrary is the ml.DeviceInfo library of worker devices.
const RemoteLibrary = "RPC"

// WorkerStatus is what a worker reports on its status port.
type WorkerStatus struct {
	Hostname string `json:"hostname"`
	Version  string `json:"version"`
	LlamaCpp string `json:"llama_cpp"`

	// Endpoint is the RPC host:port the worker listens on.
	Endpoint string `json:"endpoint"`

	// Running reports whether the RPC server process is up.
	Running bool `json:"running"`

	// Clients is the number of llama-server processes connected to the
	// first RPC server. An RPC server serves one client at a time.
	Clients int `json:"clients"`

	// MaxSlots is how many RPC servers the worker runs at most, so how many
	// models it serves at once. FreeSlots is how many more models it can take
	// now: idle RPC servers plus the ones it may still start. Slots are the
	// running RPC servers, the always running one first.
	MaxSlots  int          `json:"max_slots,omitempty"`
	FreeSlots int          `json:"free_slots"`
	Slots     []SlotStatus `json:"slots,omitempty"`

	// Transport is the RPC transport line printed by the server, e.g.
	// "TCP (RDMA auto-negotiate enabled)".
	Transport string `json:"transport,omitempty"`

	// Link is the transport the RPC server negotiated with the last
	// llama-server that connected, e.g. "RDMA (RoCEv2, rocep1s0f0, MTU 1024)"
	// or "TCP"; empty until a client connects. Transport only says what the
	// server can offer, since RDMA is negotiated per connection.
	Link string `json:"link,omitempty"`

	Devices []RemoteDevice `json:"devices"`

	// GPU is the GPU load of the worker machine (any process, not only RPC).
	GPU *GPUStats `json:"gpu,omitempty"`

	StartedAt time.Time `json:"started_at"`
	Restarts  int       `json:"restarts,omitempty"`
	LastError string    `json:"last_error,omitempty"`
}

// SlotStatus is one RPC server of a worker.
type SlotStatus struct {
	Index    int    `json:"index"`
	Endpoint string `json:"endpoint"`
	Running  bool   `json:"running"`
	Clients  int    `json:"clients"`

	// Reserved is set while a llama-server that got the slot has not
	// connected yet.
	Reserved bool `json:"reserved,omitempty"`

	// IdleSince is when the last client of an extra slot left; the worker
	// stops the slot after the slot idle time.
	IdleSince time.Time `json:"idle_since,omitzero"`

	// Link is the transport negotiated with the last client of this slot.
	Link string `json:"link,omitempty"`
}

// Free reports whether the slot can take a model now.
func (s SlotStatus) Free() bool {
	return s.Running && s.Clients == 0 && !s.Reserved
}

// SlotsLabel says how many slots are in use, e.g. "1 of 3 slots used".
func (s *WorkerStatus) SlotsLabel() string {
	if s == nil || s.MaxSlots == 0 {
		return ""
	}
	return fmt.Sprintf("%d of %d slots used", s.MaxSlots-s.FreeSlots, s.MaxSlots)
}

// TransportLabel is the transport to show for the worker: the one negotiated
// with the last client, or what the server offers while none has connected.
func (s *WorkerStatus) TransportLabel() string {
	link := ""
	if s != nil {
		link = s.Link
		for _, sl := range s.Slots {
			if link == "" {
				link = sl.Link
			}
		}
	}
	switch {
	case s == nil:
		return ""
	case link != "":
		return link
	case s.Transport != "":
		return s.Transport + ", no client yet"
	}
	return ""
}

// RemoteDevice is a device exposed by a worker.
type RemoteDevice struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	TotalMemory uint64 `json:"total_memory"`
	FreeMemory  uint64 `json:"free_memory"`
	Integrated  bool   `json:"integrated,omitempty"`
}

// WorkerCheck is the result of checking one worker before using it.
type WorkerCheck struct {
	Worker Worker        `json:"worker"`
	Status *WorkerStatus `json:"status,omitempty"`

	// Usable is true when the worker answers, runs the same build and its
	// RPC server is up.
	Usable bool `json:"usable"`

	// Busy is true when the worker cannot take another model: all its RPC
	// servers serve one.
	Busy bool `json:"busy"`

	// Problem is a one sentence explanation when the worker is not usable,
	// and Fix the command that repairs it.
	Problem string `json:"problem,omitempty"`
	Fix     string `json:"fix,omitempty"`
}

const probeTimeout = 1500 * time.Millisecond

var httpClient = &http.Client{
	Timeout: probeTimeout,
	Transport: &http.Transport{
		Proxy:             nil,
		DisableKeepAlives: true,
	},
}

// ProbeWorker fetches the status of a worker.
func ProbeWorker(ctx context.Context, w Worker) (*WorkerStatus, error) {
	ctx, cancel := context.WithTimeout(ctx, probeTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://"+w.StatusAddr()+"/v1/status", nil)
	if err != nil {
		return nil, err
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("worker status: %s", resp.Status)
	}

	var st WorkerStatus
	if err := json.NewDecoder(resp.Body).Decode(&st); err != nil {
		return nil, fmt.Errorf("worker status: %w", err)
	}
	return &st, nil
}

// CheckWorker probes a worker and decides whether this server can use it.
func CheckWorker(ctx context.Context, w Worker) WorkerCheck {
	c := WorkerCheck{Worker: w}

	st, err := ProbeWorker(ctx, w)
	if err != nil {
		c.Problem = fmt.Sprintf("worker %s does not answer on %s (%v). Check that it is powered on and the direct cable is connected.", w.Label(), w.StatusAddr(), err)
		c.Fix = "ollama cluster doctor"
		return c
	}
	c.Status = st

	switch {
	case st.LlamaCpp != LlamaCppVersion:
		c.Problem = fmt.Sprintf("worker %s runs llama.cpp %s, this Ollama needs %s.", w.Label(), st.LlamaCpp, LlamaCppVersion)
		c.Fix = "ollama cluster upgrade"
	case st.Version != version.Version:
		c.Problem = fmt.Sprintf("worker %s runs Ollama %s, this Ollama is %s.", w.Label(), st.Version, version.Version)
		c.Fix = "ollama cluster upgrade"
	case !st.Running:
		c.Problem = fmt.Sprintf("worker %s answers, but its RPC server is not running: %s", w.Label(), st.LastError)
		c.Fix = "ollama cluster doctor"
	case len(st.Devices) == 0:
		c.Problem = fmt.Sprintf("worker %s reports no GPU.", w.Label())
		c.Fix = "ollama cluster doctor"
	default:
		c.Usable = true
		c.Busy = st.busy()
	}
	return c
}

// CheckWorkers checks all active workers in parallel.
func CheckWorkers(ctx context.Context, cfg *Config) []WorkerCheck {
	workers := cfg.ActiveWorkers()
	checks := make([]WorkerCheck, len(workers))
	var wg sync.WaitGroup
	for i, w := range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			checks[i] = CheckWorker(ctx, w)
		}()
	}
	wg.Wait()
	return checks
}

// busy reports whether the worker has no RPC server left for a model.
func (st *WorkerStatus) busy() bool {
	if st.MaxSlots == 0 {
		// a worker without slots runs one RPC server
		return st.Clients > 0
	}
	return st.FreeSlots == 0
}

// RPCDeviceName is the name llama-server gives the n-th device of a worker
// in its logs, e.g. "RPC0[10.100.48.2:50052]".
func RPCDeviceName(endpoint string, n int) string {
	return fmt.Sprintf("RPC%d[%s]", n, endpoint)
}

// DeviceInfos turns a usable worker into scheduler devices. A busy worker is
// reported with no free memory so the scheduler does not give it another
// model; an RPC server blocks a second client until the first one
// disconnects. The device IDs name the first RPC server of the worker
// whichever slot a model gets, so the scheduler adds up the models of all
// slots on the same GPU.
func (c WorkerCheck) DeviceInfos() []ml.DeviceInfo {
	if !c.Usable || c.Status == nil {
		return nil
	}

	endpoint := c.Worker.Endpoint()
	devices := make([]ml.DeviceInfo, 0, len(c.Status.Devices))
	for i, d := range c.Status.Devices {
		free := d.FreeMemory
		if c.Busy {
			free = 0
		}
		devices = append(devices, ml.DeviceInfo{
			DeviceID: ml.DeviceID{
				ID:      fmt.Sprintf("%s/%d", endpoint, i),
				Library: RemoteLibrary,
			},
			Name:        RPCDeviceName(endpoint, i),
			Description: fmt.Sprintf("%s on %s", d.Description, c.Worker.Label()),
			TotalMemory: d.TotalMemory,
			FreeMemory:  free,
			LibraryPath: []string{ml.LibOllamaPath},
		})
	}
	return devices
}

var (
	stateMu   sync.Mutex
	lastState = map[string]string{}
)

// logTransition logs a worker state only when it changes, so the device
// refresh before every model load does not flood the log.
func logTransition(c WorkerCheck) {
	state := "usable"
	if !c.Usable {
		state = c.Problem
	} else if c.Busy {
		state = "busy"
	}

	key := c.Worker.Endpoint()
	stateMu.Lock()
	prev, seen := lastState[key]
	lastState[key] = state
	stateMu.Unlock()
	if seen && prev == state {
		return
	}

	switch {
	case !c.Usable:
		slog.Warn("cluster worker skipped, models load on this machine only", "worker", c.Worker.Label(), "problem", c.Problem, "fix", c.Fix)
	case c.Busy:
		slog.Info("cluster worker busy, all its RPC servers hold a model", "worker", c.Worker.Label(), "endpoint", key)
	default:
		var total, free uint64
		for _, d := range c.Status.Devices {
			total += d.TotalMemory
			free += d.FreeMemory
		}
		slog.Info("cluster worker available", "worker", c.Worker.Label(), "endpoint", key,
			"transport", c.Status.TransportLabel(), "total", format.HumanBytes2(total), "available", format.HumanBytes2(free))
	}
}

// Devices returns the devices of all usable workers. It returns nothing when
// no cluster is configured.
func Devices(ctx context.Context) []ml.DeviceInfo {
	cfg, err := LoadConfig()
	if err != nil {
		slog.Warn("cluster configuration unreadable, workers ignored", "path", ConfigPath(), "error", err)
		return nil
	}
	if len(cfg.ActiveWorkers()) == 0 {
		return nil
	}

	var devices []ml.DeviceInfo
	for _, c := range cachedChecks(ctx, cfg) {
		logTransition(c)
		devices = append(devices, c.DeviceInfos()...)
	}
	return devices
}

var (
	checkCacheMu sync.Mutex
	checkCache   = map[string]cachedCheck{}
)

type cachedCheck struct {
	check WorkerCheck
	at    time.Time
}

// cachedChecks reuses recent worker checks: the scheduler refreshes devices
// before every load, and a powered-off worker would otherwise add a probe
// timeout to each of them. Failures are remembered longer than successes.
func cachedChecks(ctx context.Context, cfg *Config) []WorkerCheck {
	workers := cfg.ActiveWorkers()
	checks := make([]WorkerCheck, len(workers))
	var wg sync.WaitGroup
	for i, w := range workers {
		key := w.Endpoint() + "|" + w.StatusAddr()
		checkCacheMu.Lock()
		cached, ok := checkCache[key]
		checkCacheMu.Unlock()
		ttl := time.Second
		if ok && !cached.check.Usable {
			ttl = 10 * time.Second
		}
		if ok && time.Since(cached.at) < ttl {
			checks[i] = cached.check
			continue
		}

		wg.Add(1)
		go func() {
			defer wg.Done()
			c := checkAfterRelease(ctx, w)
			checkCacheMu.Lock()
			checkCache[key] = cachedCheck{check: c, at: time.Now()}
			checkCacheMu.Unlock()
			checks[i] = c
		}()
	}
	wg.Wait()
	return checks
}

var (
	releasedMu sync.Mutex
	released   = map[string]time.Time{}
)

// releaseGrace is how long a worker may go on showing the client of a
// llama-server that has exited; the RPC server notices the closed connection
// a moment later.
const releaseGrace = 3 * time.Second

// Released records that a llama-server of this Ollama using these worker
// endpoints has exited, so the next check waits for the worker to drop the
// client instead of reporting it busy. Without it a model loaded right after
// another one unloads finds the worker busy and loads on this machine only.
func Released(endpoints []string) {
	if len(endpoints) == 0 {
		return
	}
	now := time.Now()
	releasedMu.Lock()
	for _, ep := range endpoints {
		released[ep] = now
	}
	releasedMu.Unlock()

	checkCacheMu.Lock()
	for key, c := range checkCache {
		if slices.Contains(endpoints, c.check.Worker.Endpoint()) {
			delete(checkCache, key)
		}
	}
	checkCacheMu.Unlock()
}

// checkAfterRelease checks a worker; a worker still busy shortly after a
// llama-server of this Ollama released it is checked again for up to
// releaseGrace.
func checkAfterRelease(ctx context.Context, w Worker) WorkerCheck {
	ep := w.Endpoint()
	c := CheckWorker(ctx, w)
	releasedMu.Lock()
	at, ok := released[ep]
	releasedMu.Unlock()
	if !ok {
		return c
	}
	deadline := at.Add(releaseGrace)
	for c.Usable && c.Busy && time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return c
		case <-time.After(200 * time.Millisecond):
		}
		c = CheckWorker(ctx, w)
	}
	// the release is settled once the worker is free or the grace is over
	if !c.Busy || !time.Now().Before(deadline) {
		releasedMu.Lock()
		if released[ep].Equal(at) {
			delete(released, ep)
		}
		releasedMu.Unlock()
	}
	return c
}

// IsRemote reports whether a device is a worker device.
func IsRemote(d ml.DeviceInfo) bool {
	return d.Library == RemoteLibrary
}

// SplitDevices separates local devices from worker devices.
func SplitDevices(gpus []ml.DeviceInfo) (local, remote []ml.DeviceInfo) {
	for _, g := range gpus {
		if IsRemote(g) {
			remote = append(remote, g)
		} else {
			local = append(local, g)
		}
	}
	return local, remote
}

// Endpoints returns the unique worker endpoints used by a device list.
func Endpoints(gpus []ml.DeviceInfo) []string {
	var endpoints []string
	seen := map[string]bool{}
	for _, g := range gpus {
		if !IsRemote(g) {
			continue
		}
		ep := endpointOf(g.ID)
		if !seen[ep] {
			seen[ep] = true
			endpoints = append(endpoints, ep)
		}
	}
	return endpoints
}

func endpointOf(id string) string {
	for i := len(id) - 1; i >= 0; i-- {
		if id[i] == '/' {
			return id[:i]
		}
	}
	return id
}

// ErrWorkerBusy means all RPC servers of a worker serve other llama-servers.
var ErrWorkerBusy = errors.New("busy with other models, no free RPC slot")

// VerifyEndpoints re-checks the workers behind endpoints right before
// llama-server starts. It returns the reason when one of them can no longer
// take the model.
func VerifyEndpoints(ctx context.Context, endpoints []string) error {
	if len(endpoints) == 0 {
		return nil
	}
	cfg, err := LoadConfig()
	if err != nil {
		return err
	}

	byEndpoint := map[string]Worker{}
	for _, w := range cfg.ActiveWorkers() {
		byEndpoint[w.Endpoint()] = w
	}
	for _, ep := range endpoints {
		w, ok := byEndpoint[ep]
		if !ok {
			return fmt.Errorf("worker %s is no longer configured", ep)
		}
		c := CheckWorker(ctx, w)
		logTransition(c)
		if !c.Usable {
			return fmt.Errorf("%s Run: %s", c.Problem, c.Fix)
		}
		if c.Busy {
			return fmt.Errorf("worker %s: %w", w.Label(), ErrWorkerBusy)
		}
	}
	return nil
}

// LlamaDeviceNames returns the names llama-server registers for the worker
// devices among gpus, for --device. llama.cpp numbers RPC devices across all
// --rpc endpoints in their order (RPC0, RPC1, ...); "RPC0[host:port]" is only
// the name of the buffer type in its logs.
func LlamaDeviceNames(gpus []ml.DeviceInfo) []string {
	count := map[string]int{}
	for _, g := range gpus {
		if IsRemote(g) {
			ep := endpointOf(g.ID)
			count[ep] = max(count[ep], deviceIndex(g.ID)+1)
		}
	}
	offset := map[string]int{}
	n := 0
	for _, ep := range Endpoints(gpus) {
		offset[ep] = n
		n += count[ep]
	}
	var names []string
	for _, g := range gpus {
		if IsRemote(g) {
			names = append(names, fmt.Sprintf("RPC%d", offset[endpointOf(g.ID)]+deviceIndex(g.ID)))
		}
	}
	return names
}

// SlotDevices returns copies of gpus whose worker devices carry the names
// llama-server logs for them on the reserved slots (slots maps the first RPC
// endpoint of a worker to the reserved one): "RPC0[10.100.48.2:50054]". The
// device IDs stay those of the first RPC server.
func SlotDevices(gpus []ml.DeviceInfo, slots map[string]string) []ml.DeviceInfo {
	gpus = slices.Clone(gpus)
	for i, g := range gpus {
		if !IsRemote(g) {
			continue
		}
		ep := endpointOf(g.ID)
		if slot, ok := slots[ep]; ok {
			ep = slot
		}
		gpus[i].Name = RPCDeviceName(ep, deviceIndex(g.ID))
	}
	return gpus
}

func deviceIndex(id string) int {
	ep := endpointOf(id)
	if len(ep) >= len(id) {
		return 0
	}
	i, _ := strconv.Atoi(id[len(ep)+1:])
	return i
}
