// Package worker runs the cluster worker: the ggml RPC server from Ollama's
// own llama.cpp build plus a small status server that tells the main Ollama
// which build the worker runs, how much memory it has and whether a model is
// already using it.
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
}

// Worker supervises the RPC server process.
type Worker struct {
	opts Options

	mu     sync.Mutex
	status cluster.WorkerStatus
}

// Run starts the RPC server and the status server and blocks until ctx ends.
func Run(ctx context.Context, opts Options) error {
	if opts.Addr == "" {
		return errors.New("worker address is required (the direct-link address of this machine)")
	}
	if ip := net.ParseIP(opts.Addr); ip != nil && ip.IsUnspecified() {
		return errors.New("refusing to listen on all interfaces: the RPC server has no authentication, use the direct-link address")
	}
	if opts.RPCPort == 0 {
		opts.RPCPort = cluster.DefaultRPCPort
	}
	if opts.StatusPort == 0 {
		opts.StatusPort = cluster.DefaultStatusPort
	}

	hostname, _ := os.Hostname()
	w := &Worker{opts: opts}
	w.status = cluster.WorkerStatus{
		Hostname:  hostname,
		Version:   version.Version,
		LlamaCpp:  cluster.LlamaCppVersion,
		Endpoint:  net.JoinHostPort(opts.Addr, strconv.Itoa(opts.RPCPort)),
		StartedAt: time.Now(),
	}

	ln, err := net.Listen("tcp", net.JoinHostPort(opts.Addr, strconv.Itoa(opts.StatusPort)))
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
		"rpc", w.status.Endpoint, "status", ln.Addr().String())
	return w.supervise(ctx)
}

// supervise keeps the RPC server running, restarting it with backoff.
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

func (w *Worker) runOnce(ctx context.Context) error {
	exe, err := llm.FindLlamaCppBinary("ggml-rpc-server")
	if err != nil {
		return fmt.Errorf("this Ollama build has no RPC worker: %w", err)
	}
	libDirs := []string{ml.LibOllamaPath}
	if dir := gpuLibDir(); dir != "" {
		libDirs = append(libDirs, dir)
	}

	devices, err := listDevices(ctx, libDirs)
	if err != nil {
		return err
	}
	if len(devices) == 0 {
		return errors.New("no GPU found by the RPC backend (check the NVIDIA driver and --gpus all)")
	}
	markIntegrated(devices)
	w.mu.Lock()
	w.status.Devices = devices
	w.mu.Unlock()

	args := []string{"-H", w.opts.Addr, "-p", strconv.Itoa(w.opts.RPCPort)}
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

	slog.Info("starting rpc server", "cmd", cmd.String(), "libs", libDirs)
	if err := cmd.Start(); err != nil {
		return err
	}

	lastLine := w.readOutput(stdout)
	err = cmd.Wait()
	if err == nil {
		err = errors.New("rpc server exited")
	}
	if lastLine != "" {
		err = fmt.Errorf("%w: %s", err, lastLine)
	}
	return err
}

// readOutput forwards the RPC server output to the log and records the
// transport it reports. It returns the last non-empty line.
func (w *Worker) readOutput(r io.Reader) (lastLine string) {
	var link linkState
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) != "" {
			lastLine = strings.TrimSpace(line)
		}
		fmt.Fprintln(os.Stderr, line)

		if key, value, ok := strings.Cut(line, ":"); ok && strings.TrimSpace(key) == "transport" {
			w.mu.Lock()
			w.status.Transport = strings.TrimSpace(value)
			w.mu.Unlock()
		}
		if link.update(line) {
			w.mu.Lock()
			w.status.Link = link.label
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
	return mux
}

func (w *Worker) snapshot() cluster.WorkerStatus {
	w.mu.Lock()
	st := w.status
	st.Devices = slices.Clone(w.status.Devices)
	w.mu.Unlock()

	listening, clients := rpcSockets(w.opts.RPCPort)
	st.Running = listening
	st.Clients = clients
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
