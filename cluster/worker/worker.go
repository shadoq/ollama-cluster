// Package worker runs the cluster worker: the ggml RPC servers from Ollama's
// own llama.cpp build plus a small status server that tells the main Ollama
// which build the worker runs, how much memory it has and which RPC servers
// are free. An RPC server serves one model at a time, so the worker runs one
// per model (a slot): the first one always, the others on request from the
// main Ollama, up to MaxSlots.
package worker

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/cluster"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/version"
)

// Options configure a worker.
type Options struct {
	// Addr is the address to listen on; it must be the direct-link address.
	Addr       string
	RPCPort    int
	StatusPort int

	// Cache enables the RPC server tensor cache (-c), which can shorten
	// repeated loads of the same model.
	Cache bool

	// Threads is the CPU thread count passed to the RPC server (0 = default).
	Threads int

	// MaxSlots is how many RPC servers run at most (0 = default).
	MaxSlots int

	// SlotIdle is how long a released extra RPC server is kept (0 = default).
	SlotIdle time.Duration
}

// Worker supervises the RPC server processes.
type Worker struct {
	opts  Options
	ports []int

	mu     sync.Mutex
	status cluster.WorkerStatus
	// slots are the RPC servers that run or start, ordered by index; the
	// first one is always there.
	slots []*slot

	// run runs the RPC server of a slot until it exits or ctx ends; sockets
	// reports whether a port listens and how many clients it has. Tests
	// replace them.
	run     func(ctx context.Context, s *slot) error
	sockets func(port int) (listening bool, clients int)
	now     func() time.Time
}

// slot is one RPC server.
type slot struct {
	index int
	port  int

	// link is the transport negotiated with the last client.
	link string
	// reservedUntil keeps the slot for the llama-server that got it until
	// that one connects.
	reservedUntil time.Time
	// idleSince is when an extra slot was first seen without a client.
	idleSince time.Time
	stopping  bool

	cancel context.CancelFunc
}

// errNoSlot means every slot serves a model and no more may start.
var errNoSlot = errors.New("all RPC servers of the worker serve a model")

func newWorker(opts Options) *Worker {
	if opts.RPCPort == 0 {
		opts.RPCPort = cluster.DefaultRPCPort
	}
	if opts.StatusPort == 0 {
		opts.StatusPort = cluster.DefaultStatusPort
	}
	if opts.MaxSlots <= 0 {
		opts.MaxSlots = cluster.DefaultMaxSlots
	}
	opts.MaxSlots = min(opts.MaxSlots, cluster.MaxSlotLimit)
	if opts.SlotIdle <= 0 {
		opts.SlotIdle = cluster.DefaultSlotIdle
	}
	w := &Worker{
		opts:    opts,
		ports:   cluster.SlotPorts(opts.RPCPort, opts.StatusPort, opts.MaxSlots),
		sockets: rpcSockets,
		now:     time.Now,
	}
	w.run = w.runServer
	w.slots = []*slot{{index: 0, port: opts.RPCPort}}
	return w
}

// Run starts the RPC server and the status server and blocks until ctx ends.
func Run(ctx context.Context, opts Options) error {
	if opts.Addr == "" {
		return errors.New("worker address is required (the direct-link address of this machine)")
	}
	if ip := net.ParseIP(opts.Addr); ip != nil && ip.IsUnspecified() {
		return errors.New("refusing to listen on all interfaces: the RPC server has no authentication, use the direct-link address")
	}

	hostname, _ := os.Hostname()
	w := newWorker(opts)
	w.status = cluster.WorkerStatus{
		Hostname:  hostname,
		Version:   version.Version,
		LlamaCpp:  cluster.LlamaCppVersion,
		Endpoint:  w.endpoint(w.opts.RPCPort),
		StartedAt: time.Now(),
	}

	ln, err := net.Listen("tcp", net.JoinHostPort(w.opts.Addr, strconv.Itoa(w.opts.StatusPort)))
	if err != nil {
		return fmt.Errorf("status server: %w", err)
	}
	srv := &http.Server{Handler: w.handler(), ReadHeaderTimeout: 5 * time.Second}
	go func() {
		<-ctx.Done()
		_ = srv.Close()
	}()
	go func() {
		if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			slog.Error("status server stopped", "error", err)
		}
	}()

	slog.Info("cluster worker starting", "version", version.Version, "llama_cpp", cluster.LlamaCppVersion,
		"rpc", w.status.Endpoint, "status", ln.Addr().String(), "max_slots", w.opts.MaxSlots, "slot_idle", w.opts.SlotIdle)
	go w.reapLoop(ctx)
	return w.supervise(ctx)
}

// supervise keeps the first RPC server running, restarting it with backoff.
func (w *Worker) supervise(ctx context.Context) error {
	backoff := 2 * time.Second
	for {
		started := time.Now()
		err := w.runOnce(ctx)
		if ctx.Err() != nil {
			return nil
		}

		w.mu.Lock()
		w.status.Restarts++
		if err != nil {
			w.status.LastError = err.Error()
		}
		w.mu.Unlock()

		if time.Since(started) > time.Minute {
			backoff = 2 * time.Second
		}
		slog.Warn("rpc server exited, restarting", "error", err, "in", backoff)
		select {
		case <-ctx.Done():
			return nil
		case <-time.After(backoff):
		}
		backoff = min(backoff*2, 30*time.Second)
	}
}

// runOnce lists the devices and runs the first RPC server until it exits.
func (w *Worker) runOnce(ctx context.Context) error {
	devices, err := listDevices(ctx, rpcLibDirs())
	if err != nil {
		return err
	}
	if len(devices) == 0 {
		return errors.New("no GPU found by the RPC backend (check the NVIDIA driver and --gpus all)")
	}
	markIntegrated(devices)
	w.mu.Lock()
	w.status.Devices = devices
	first := w.slots[0]
	w.mu.Unlock()

	return w.run(ctx, first)
}

func rpcLibDirs() []string {
	libDirs := []string{ml.LibOllamaPath}
	if dir := gpuLibDir(); dir != "" {
		libDirs = append(libDirs, dir)
	}
	return libDirs
}

// runServer runs the RPC server of a slot until it exits or ctx ends.
func (w *Worker) runServer(ctx context.Context, s *slot) error {
	exe, err := llm.FindLlamaCppBinary("ggml-rpc-server")
	if err != nil {
		return fmt.Errorf("this Ollama build has no RPC worker: %w", err)
	}
	libDirs := rpcLibDirs()

	args := []string{"-H", w.opts.Addr, "-p", strconv.Itoa(s.port)}
	if w.opts.Cache {
		args = append(args, "-c")
	}
	if w.opts.Threads > 0 {
		args = append(args, "-t", strconv.Itoa(w.opts.Threads))
	}

	// The RPC server prints with printf; line buffering keeps its log and
	// transport line timely when stdout is a pipe.
	name, cmdArgs := exe, args
	if stdbuf, err := exec.LookPath("stdbuf"); err == nil {
		name, cmdArgs = stdbuf, append([]string{"-oL", exe}, args...)
	}
	cmd := exec.CommandContext(ctx, name, cmdArgs...)
	cmd.WaitDelay = 5 * time.Second
	llm.SetupLlamaServerCommandEnv(cmd, exe, libDirs, nil)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	cmd.Stderr = cmd.Stdout

	slog.Info("starting rpc server", "slot", s.index, "cmd", cmd.String(), "libs", libDirs)
	if err := cmd.Start(); err != nil {
		return err
	}

	lastLine := w.readOutput(stdout, s)
	err = cmd.Wait()
	if err == nil {
		err = errors.New("rpc server exited")
	}
	if lastLine != "" {
		err = fmt.Errorf("%w: %s", err, lastLine)
	}
	return err
}

// readOutput forwards the RPC server output of a slot to the log and records
// the transport it reports. It returns the last non-empty line.
func (w *Worker) readOutput(r io.Reader, s *slot) (lastLine string) {
	prefix := ""
	if s.index > 0 {
		prefix = fmt.Sprintf("[slot %d] ", s.index)
	}
	var link linkState
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) != "" {
			lastLine = strings.TrimSpace(line)
		}
		fmt.Fprintln(os.Stderr, prefix+line)

		if key, value, ok := strings.Cut(line, ":"); ok && strings.TrimSpace(key) == "transport" {
			w.mu.Lock()
			w.status.Transport = strings.TrimSpace(value)
			w.mu.Unlock()
		}
		if link.update(line) {
			w.mu.Lock()
			s.link = link.label
			if s.index == 0 {
				w.status.Link = link.label
			}
			w.mu.Unlock()
		}
	}
	return lastLine
}

// rdmaProbed matches the RPC server line printed when a client offers RDMA:
//
//	RDMA probed: dev=rocep1s0f0 gid=3 RoCEv2 qpn=4812 inline=316
var rdmaProbed = regexp.MustCompile(`^RDMA probed: dev=(\S+) gid=\d+\s*(\S*)\s+qpn=`)

// rdmaActivated matches the line printed when the connection switched to RDMA:
//
//	RDMA activated: qpn=4812->4927 mtu=1024 rx_depth=24
var rdmaActivated = regexp.MustCompile(`^RDMA activated: .*\bmtu=(\d+)`)

// linkState follows the per-connection transport negotiation in the RPC
// server output. A connection starts on TCP and moves to RDMA only when the
// server prints "RDMA activated".
type linkState struct {
	dev, mode string
	label     string
}

// update reads one output line and reports whether the label changed.
func (l *linkState) update(line string) bool {
	line = strings.TrimSpace(line)
	before := l.label
	switch {
	case line == "Accepted client connection":
		l.dev, l.mode = "", ""
		l.label = "TCP"
	case rdmaProbed.MatchString(line):
		m := rdmaProbed.FindStringSubmatch(line)
		l.dev, l.mode = m[1], m[2]
	case rdmaActivated.MatchString(line):
		m := rdmaActivated.FindStringSubmatch(line)
		parts := []string{}
		if l.mode != "" {
			parts = append(parts, l.mode)
		}
		if l.dev != "" {
			parts = append(parts, l.dev)
		}
		parts = append(parts, "MTU "+m[1])
		l.label = "RDMA (" + strings.Join(parts, ", ") + ")"
	case strings.HasPrefix(line, "RDMA activate failed"):
		l.label = "TCP (RDMA activation failed)"
	}
	return l.label != before
}

// deviceLine matches llama-server --list-devices output:
//
//	CUDA0: NVIDIA GB10 (122502 MiB, 117431 MiB free)
var deviceLine = regexp.MustCompile(`^\s+(\S+):\s+(.+?)\s+\((\d+)\s+MiB,\s+(\d+)\s+MiB\s+free\)`)

// listDevices asks llama-server from the same build which GPUs the backends
// see; the RPC server exposes the same non-CPU devices.
func listDevices(ctx context.Context, libDirs []string) ([]cluster.RemoteDevice, error) {
	exe, err := llm.FindLlamaServer()
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, exe, "--list-devices", "--offline")
	llm.SetupLlamaServerCommandEnv(cmd, exe, libDirs, nil)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("listing devices: %w: %s", err, lastNonEmpty(string(out)))
	}
	return parseDevices(string(out)), nil
}

func parseDevices(out string) []cluster.RemoteDevice {
	var devices []cluster.RemoteDevice
	for _, line := range strings.Split(out, "\n") {
		m := deviceLine.FindStringSubmatch(line)
		if m == nil || strings.HasPrefix(m[1], "CPU") || strings.HasPrefix(m[1], "BLAS") {
			continue
		}
		total, _ := strconv.ParseUint(m[3], 10, 64)
		free, _ := strconv.ParseUint(m[4], 10, 64)
		if total == 0 {
			continue
		}
		devices = append(devices, cluster.RemoteDevice{
			Name:        m[1],
			Description: m[2],
			TotalMemory: total << 20,
			FreeMemory:  free << 20,
		})
	}
	return devices
}

func lastNonEmpty(s string) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	return strings.TrimSpace(lines[len(lines)-1])
}

// markIntegrated flags devices whose memory is the system memory (GB10 and
// other unified-memory GPUs). Their free memory is read live from the system.
func markIntegrated(devices []cluster.RemoteDevice) {
	total, _, ok := cluster.SystemMemory()
	if !ok {
		return
	}
	for i := range devices {
		d := devices[i].TotalMemory
		if d > total*85/100 && d < total*115/100 {
			devices[i].Integrated = true
		}
	}
}

func (w *Worker) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/status", func(rw http.ResponseWriter, r *http.Request) {
		rw.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(rw).Encode(w.snapshot())
	})
	mux.HandleFunc("POST /v1/slots/acquire", func(rw http.ResponseWriter, r *http.Request) {
		lease, err := w.acquire(r.Context())
		switch {
		case errors.Is(err, errNoSlot):
			http.Error(rw, err.Error(), http.StatusConflict)
			return
		case err != nil:
			http.Error(rw, err.Error(), http.StatusServiceUnavailable)
			return
		}
		rw.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(rw).Encode(lease)
	})
	mux.HandleFunc("POST /v1/slots/release", func(rw http.ResponseWriter, r *http.Request) {
		var req cluster.SlotRelease
		if err := json.NewDecoder(io.LimitReader(r.Body, 4096)).Decode(&req); err != nil {
			http.Error(rw, err.Error(), http.StatusBadRequest)
			return
		}
		w.release(req.Endpoint)
		rw.WriteHeader(http.StatusNoContent)
	})
	return mux
}

func (w *Worker) endpoint(port int) string {
	return net.JoinHostPort(w.opts.Addr, strconv.Itoa(port))
}

// slotState checks the sockets of a slot and settles its reservation, which
// ends when its llama-server connects or the reservation runs out. w.mu must
// be held.
func (w *Worker) slotState(s *slot, now time.Time) cluster.SlotStatus {
	listening, clients := w.sockets(s.port)
	if clients > 0 || now.After(s.reservedUntil) {
		s.reservedUntil = time.Time{}
	}
	st := cluster.SlotStatus{
		Index:    s.index,
		Endpoint: w.endpoint(s.port),
		Running:  listening && !s.stopping,
		Clients:  clients,
		Reserved: !s.reservedUntil.IsZero(),
		Link:     s.link,
	}
	if s.index > 0 && clients == 0 {
		st.IdleSince = s.idleSince
	}
	return st
}

// acquire reserves a free RPC server for a model, starting a new one when
// none is free and fewer than MaxSlots run.
func (w *Worker) acquire(ctx context.Context) (cluster.SlotLease, error) {
	w.mu.Lock()
	now := w.now()
	for _, s := range w.slots {
		if w.slotState(s, now).Free() {
			s.reservedUntil = now.Add(cluster.SlotLeaseTTL)
			s.idleSince = time.Time{}
			w.mu.Unlock()
			slog.Info("rpc slot reserved", "slot", s.index, "port", s.port)
			return cluster.SlotLease{Index: s.index, Endpoint: w.endpoint(s.port)}, nil
		}
	}
	if len(w.slots) >= w.opts.MaxSlots {
		w.mu.Unlock()
		return cluster.SlotLease{}, errNoSlot
	}

	// the lowest unused index; the new slot stays reserved while it starts
	index := 0
	for _, s := range w.slots {
		if s.index == index {
			index++
		}
	}
	sctx, cancel := context.WithCancel(context.Background())
	s := &slot{index: index, port: w.ports[index], reservedUntil: now.Add(cluster.SlotStartTimeout + cluster.SlotLeaseTTL), cancel: cancel}
	w.slots = append(w.slots, s)
	slices.SortFunc(w.slots, func(a, b *slot) int { return a.index - b.index })
	w.mu.Unlock()

	done := make(chan error, 1)
	go func() {
		err := w.run(sctx, s)
		w.mu.Lock()
		w.slots = slices.DeleteFunc(w.slots, func(o *slot) bool { return o == s })
		w.mu.Unlock()
		if sctx.Err() == nil {
			slog.Warn("rpc slot exited", "slot", s.index, "port", s.port, "error", err)
		}
		cancel()
		done <- err
	}()

	slog.Info("starting rpc slot", "slot", index, "port", s.port)
	deadline := time.NewTimer(cluster.SlotStartTimeout)
	defer deadline.Stop()
	tick := time.NewTicker(100 * time.Millisecond)
	defer tick.Stop()
	for {
		w.mu.Lock()
		listening, _ := w.sockets(s.port)
		if listening {
			s.reservedUntil = w.now().Add(cluster.SlotLeaseTTL)
			w.mu.Unlock()
			slog.Info("rpc slot reserved", "slot", s.index, "port", s.port)
			return cluster.SlotLease{Index: s.index, Endpoint: w.endpoint(s.port)}, nil
		}
		w.mu.Unlock()
		select {
		case err := <-done:
			if err == nil {
				err = errors.New("rpc server exited")
			}
			return cluster.SlotLease{}, fmt.Errorf("starting rpc slot %d: %w", index, err)
		case <-deadline.C:
			cancel()
			return cluster.SlotLease{}, fmt.Errorf("rpc slot %d did not listen on port %d within %s", index, s.port, cluster.SlotStartTimeout)
		case <-ctx.Done():
			// the new slot goes on starting and serves the next request
			return cluster.SlotLease{}, ctx.Err()
		case <-tick.C:
		}
	}
}

// release ends the reservation of the slot at endpoint.
func (w *Worker) release(endpoint string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	for _, s := range w.slots {
		if w.endpoint(s.port) == endpoint && !s.reservedUntil.IsZero() {
			s.reservedUntil = time.Time{}
			slog.Info("rpc slot released", "slot", s.index, "port", s.port)
		}
	}
}

// reap stops extra slots that had neither a client nor a reservation for
// the slot idle time. The first slot always runs.
func (w *Worker) reap() {
	w.mu.Lock()
	defer w.mu.Unlock()
	now := w.now()
	for _, s := range w.slots {
		if s.index == 0 || s.stopping {
			continue
		}
		st := w.slotState(s, now)
		switch {
		case st.Clients > 0 || st.Reserved:
			s.idleSince = time.Time{}
		case s.idleSince.IsZero():
			s.idleSince = now
		case now.Sub(s.idleSince) >= w.opts.SlotIdle:
			slog.Info("stopping idle rpc slot", "slot", s.index, "port", s.port, "idle", now.Sub(s.idleSince).Round(time.Second))
			s.stopping = true
			s.cancel()
		}
	}
}

func (w *Worker) reapLoop(ctx context.Context) {
	t := time.NewTicker(15 * time.Second)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			w.mu.Lock()
			for _, s := range w.slots {
				if s.cancel != nil {
					s.cancel()
				}
			}
			w.mu.Unlock()
			return
		case <-t.C:
			w.reap()
		}
	}
}

func (w *Worker) snapshot() cluster.WorkerStatus {
	w.mu.Lock()
	st := w.status
	st.Devices = slices.Clone(w.status.Devices)
	now := w.now()
	st.MaxSlots = w.opts.MaxSlots
	st.FreeSlots = w.opts.MaxSlots - len(w.slots)
	st.Slots = nil
	for _, s := range w.slots {
		ss := w.slotState(s, now)
		if ss.Free() {
			st.FreeSlots++
		}
		if s.index == 0 {
			st.Running = ss.Running
			st.Clients = ss.Clients
		}
		st.Slots = append(st.Slots, ss)
	}
	w.mu.Unlock()

	if st.Running {
		st.LastError = ""
	}
	if _, avail, ok := cluster.SystemMemory(); ok {
		for i := range st.Devices {
			if st.Devices[i].Integrated {
				st.Devices[i].FreeMemory = min(avail, st.Devices[i].TotalMemory)
			}
		}
	}
	st.GPU = cluster.ReadGPU(context.Background())
	return st
}

// gpuLibDir picks the GPU backend directory the RPC server should load, the
// same way discovery does on the main node: OLLAMA_LLM_LIBRARY if set, else
// the newest CUDA runner present.
func gpuLibDir() string {
	if lib := os.Getenv("OLLAMA_LLM_LIBRARY"); lib != "" {
		return filepath.Join(ml.LibOllamaPath, lib)
	}
	for _, name := range []string{"cuda_v13", "cuda_v12", "rocm_v7_2", "vulkan"} {
		dir := filepath.Join(ml.LibOllamaPath, name)
		if matches, _ := filepath.Glob(filepath.Join(dir, "libggml-*")); len(matches) > 0 {
			return dir
		}
	}
	return ""
}

// rpcSockets reads /proc/net/tcp{,6} and reports whether the RPC port is
// listening and how many clients are connected to it. It reports false and 0
// where /proc is unavailable.
func rpcSockets(port int) (listening bool, clients int) {
	want := fmt.Sprintf(":%04X", port)
	for _, f := range []string{"/proc/net/tcp", "/proc/net/tcp6"} {
		data, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		l, c := parseProcNetTCP(string(data), want)
		listening = listening || l
		clients += c
	}
	return listening, clients
}

func parseProcNetTCP(data, portSuffix string) (listening bool, established int) {
	lines := strings.Split(data, "\n")
	if len(lines) > 0 {
		lines = lines[1:]
	}
	for _, line := range lines {
		// sl local_address rem_address st ...; st 0A = LISTEN, 01 = ESTABLISHED
		fields := strings.Fields(line)
		if len(fields) < 4 || !strings.HasSuffix(fields[1], portSuffix) {
			continue
		}
		switch fields[3] {
		case "0A":
			listening = true
		case "01":
			established++
		}
	}
	return listening, established
}
