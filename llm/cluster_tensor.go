package llm

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cluster"
)

// Tensor parallelism across this machine and a cluster worker
// (llama.cpp -sm tensor over RPC): llama.cpp pairs two RPC servers that
// exchange partial results directly, so the GPU of this machine is served by
// a local RPC server of its own, started next to llama-server and stopped
// with it. The local server listens on this machine's address on the direct
// link: through 127.0.0.1 the connection stays on TCP and generation is
// several times slower than on one machine.

// tensorPeer is the local RPC server of one llama-server.
type tensorPeer struct {
	cmd      *exec.Cmd
	endpoint string
	done     chan struct{}
}

var (
	tensorPeersMu sync.Mutex
	// tensorPeers are the running local RPC servers by llama-server PID.
	tensorPeers = map[int]*tensorPeer{}
)

// clusterTensor reports whether the placement asked for tensor parallelism.
func clusterTensor(opts api.Options) bool {
	return opts.ClusterTensor != nil && *opts.ClusterTensor
}

// TensorParallel reports whether the llama-server with this PID runs its
// model with tensor parallelism.
func TensorParallel(pid int) bool {
	tensorPeersMu.Lock()
	defer tensorPeersMu.Unlock()
	_, ok := tensorPeers[pid]
	return ok
}

// tensorDeviceArgs are the llama-server arguments for a model split by
// tensors over the worker endpoint and the local RPC server. The worker comes
// first: the first server listens for the second one on its port + 1000, and
// the worker already listens on the direct link only. Both halves are equal,
// and the fit to the memory is off because llama-server does not support it
// in this mode; the scheduler checked that each half fits.
func tensorDeviceArgs(worker, local string) []string {
	return []string{
		"--rpc", worker + "," + local,
		"--device", "RPC0,RPC1",
		"--split-mode", "tensor",
		"--tensor-split", "1,1",
		"-ngl", "999",
		"--fit", "off",
	}
}

// startTensorPeer starts the local RPC server for the GPU of this machine on
// addr and waits until it accepts connections.
func startTensorPeer(launch llamaServerLaunchConfig, addr string) (*tensorPeer, error) {
	exe, err := FindLlamaCppBinary("ggml-rpc-server")
	if err != nil {
		return nil, err
	}

	l, err := net.Listen("tcp", net.JoinHostPort(addr, "0"))
	if err != nil {
		return nil, fmt.Errorf("no free port on %s: %w", addr, err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	l.Close()
	endpoint := net.JoinHostPort(addr, strconv.Itoa(port))

	cmd := exec.Command(exe, "-H", addr, "-p", strconv.Itoa(port))
	cmd.SysProcAttr = LlamaServerSysProcAttr
	// the same libraries and visible devices as llama-server, so the server
	// exposes the GPU of the placement and nothing else
	SetupLlamaServerCommandEnv(cmd, exe, launch.gpuLibs, launch.extraEnvsForStart())
	out, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	cmd.Stderr = cmd.Stdout

	slog.Info("starting local rpc server for tensor parallelism", "cmd", cmd)
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	p := &tensorPeer{cmd: cmd, endpoint: endpoint, done: make(chan struct{})}
	var last lastLine
	go func() {
		logTensorPeer(out, &last)
		_ = cmd.Wait()
		close(p.done)
	}()

	deadline := time.Now().Add(30 * time.Second)
	for {
		select {
		case <-p.done:
			return nil, fmt.Errorf("local rpc server exited: %s", last.get())
		default:
		}
		if c, err := net.DialTimeout("tcp", endpoint, time.Second); err == nil {
			c.Close()
			return p, nil
		}
		if time.Now().After(deadline) {
			p.stop()
			return nil, fmt.Errorf("local rpc server did not start listening on %s: %s", endpoint, last.get())
		}
		time.Sleep(100 * time.Millisecond)
	}
}

// stop kills the server and waits for it to exit.
func (p *tensorPeer) stop() {
	if p == nil {
		return
	}
	if err := p.cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
		slog.Warn("stopping local rpc server", "error", err)
	}
	select {
	case <-p.done:
	case <-time.After(10 * time.Second):
		slog.Warn("local rpc server did not exit", "pid", p.cmd.Process.Pid)
	}
}

// registerTensorPeer ties a started local RPC server to its llama-server.
func registerTensorPeer(pid int, p *tensorPeer) {
	if p == nil {
		return
	}
	tensorPeersMu.Lock()
	tensorPeers[pid] = p
	tensorPeersMu.Unlock()
}

// stopTensorPeer stops the local RPC server of the llama-server with this
// PID, if it has one.
func stopTensorPeer(pid int) {
	tensorPeersMu.Lock()
	p := tensorPeers[pid]
	delete(tensorPeers, pid)
	tensorPeersMu.Unlock()
	if p != nil {
		p.stop()
		slog.Debug("local rpc server stopped", "pid", p.cmd.Process.Pid)
	}
}

// tensorBuffer is the device name llama-server logs for buffers split by
// tensors ("Meta() model buffer size = ..."). The size is what each of the
// GPUs holds, not the total: gpt-oss:120b (65 GB) logs 30663 MiB.
const tensorBuffer = "Meta()"

// tensorShare is the memory one of the two GPUs holds in split buffers for
// the llama-server with this PID when it runs with tensor parallelism; both
// get equal halves (--tensor-split 1,1).
func tensorShare(pid int, byDevice map[string]uint64) uint64 {
	if !TensorParallel(pid) {
		return 0
	}
	return byDevice[tensorBuffer]
}

// appendTensorArgs starts the local RPC server and returns the arguments for
// tensor parallelism with the worker at endpoint. When that is not possible
// it returns nil and the caller splits by layers.
func appendTensorArgs(params []string, launch llamaServerLaunchConfig, endpoint string) ([]string, *tensorPeer) {
	cfg, err := cluster.LoadConfig()
	if err != nil {
		slog.Warn("tensor parallelism not possible, splitting by layers", "error", err)
		return nil, nil
	}
	w, ok := cfg.WorkerByEndpoint(endpoint)
	if !ok || w.LocalAddr == "" {
		slog.Warn("tensor parallelism not possible, splitting by layers", "error", "no local_addr for worker "+endpoint)
		return nil, nil
	}
	p, err := startTensorPeer(launch, w.LocalAddr)
	if err != nil {
		slog.Warn("tensor parallelism not possible, splitting by layers", "error", err)
		return nil, nil
	}
	slog.Info("running model with tensor parallelism", "worker", endpoint, "local", p.endpoint)
	return append(params, tensorDeviceArgs(endpoint, p.endpoint)...), p
}

// logTensorPeer forwards the output of the local RPC server to the log and
// keeps its last line.
func logTensorPeer(r io.Reader, last *lastLine) {
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		last.set(line)
		slog.Debug("local rpc server", "line", line)
	}
}

type lastLine struct {
	mu   sync.Mutex
	line string
}

func (l *lastLine) set(s string) {
	l.mu.Lock()
	l.line = s
	l.mu.Unlock()
}

func (l *lastLine) get() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.line
}
