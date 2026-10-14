package clustercmd

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"reflect"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cluster"
)

const (
	workerContainer      = "ollama-cluster-worker"
	defaultContextLength = 32768
	clusterImageHint     = "build the cluster image (docs/cluster.md) and pass it with --image"
)

// mainOllama is the Ollama this command manages.
type mainOllama struct {
	api       *apiClient
	version   string
	container *containerInfo // nil for a native install
	status    *cluster.StatusResponse
	statusErr error
}

func detectMain(ctx context.Context, cmd *cobra.Command) (*mainOllama, error) {
	m := &mainOllama{api: newAPI()}
	v, err := m.api.version(ctx)
	if err != nil {
		return nil, fmt.Errorf("Ollama does not answer at %s: %w. Start it first (or set OLLAMA_HOST)", m.api.base, err)
	}
	m.version = v

	name, _ := cmd.Flags().GetString("container")
	m.container, err = findOllamaContainer(ctx, name)
	if err != nil {
		return nil, err
	}
	m.status, m.statusErr = m.api.status(ctx)
	if m.status != nil {
		// the fields go into SSH commands on the worker
		if err := m.status.Config.Validate(); err != nil {
			return nil, fmt.Errorf("the cluster configuration (cluster.json) is not valid: %w", err)
		}
	}
	return m, nil
}

func (m *mainOllama) where() string {
	if m.container != nil {
		return fmt.Sprintf("Docker, container %q", m.container.name())
	}
	return "native"
}

func (m *mainOllama) config() *cluster.Config {
	if m.status != nil {
		c := m.status.Config
		return &c
	}
	return &cluster.Config{}
}

// localGPU describes the GPU of this machine via nvidia-smi.
func localGPU(ctx context.Context) (name, driver, cuda string) {
	out, err := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader").Output()
	if err == nil {
		line, _, _ := strings.Cut(strings.TrimSpace(string(out)), "\n")
		n, d, _ := strings.Cut(line, ",")
		name, driver = strings.TrimSpace(n), strings.TrimSpace(d)
	}
	if out, err := exec.CommandContext(ctx, "nvidia-smi").Output(); err == nil {
		cuda = cudaVersionFromSMI(string(out))
	}
	return name, driver, cuda
}

func cudaVersionFromSMI(out string) string {
	_, rest, ok := strings.Cut(out, "CUDA Version:")
	if !ok {
		return ""
	}
	f := strings.Fields(rest)
	if len(f) == 0 {
		return ""
	}
	return f[0]
}

// --- init -------------------------------------------------------------------

func initCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "init",
		Short: "Check this machine before attaching a second one (changes nothing)",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			r := &report{}
			m, err := detectMain(ctx, cmd)
			if err != nil {
				r.problem("Ollama", "does not answer", err.Error(), "start Ollama")
				return errors.New("fix the problem above and run 'ollama cluster init' again")
			}

			r.ok("Ollama", fmt.Sprintf("%s (%s)", m.version, m.where()))
			switch {
			case m.status != nil:
				r.ok("llama.cpp", fmt.Sprintf("%s (from the Ollama build)", m.status.LlamaCpp))
				r.ok("Cluster", "this Ollama build supports a second machine")
			case errors.Is(m.statusErr, errNoClusterAPI):
				r.warn("Cluster", "this Ollama build has no cluster support",
					"The official image cannot use a second machine.", "ollama cluster add <host> --image <cluster image>; "+clusterImageHint)
			default:
				r.problem("Cluster", "status unavailable", m.statusErr.Error(), "ollama cluster doctor")
			}

			if name, driver, cuda := localGPU(ctx); name != "" {
				r.ok("GPU", fmt.Sprintf("%s, driver %s, CUDA %s", name, driver, cuda))
			} else {
				r.problem("GPU", "not found", "nvidia-smi does not answer on this machine.", "check the NVIDIA driver")
			}

			links := slices.DeleteFunc(localLinkInterfaces(), func(i netIf) bool { return i.RDMADev == "" })
			if len(links) > 0 {
				var parts []string
				for _, l := range links {
					parts = append(parts, fmt.Sprintf("%s (%s, RDMA)", l.Addr, l.RDMADev))
				}
				r.ok("Direct cable", strings.Join(parts, ", "))
			} else {
				r.problem("Direct cable", "not found",
					"No network port with RDMA has an IPv4 address. Connect the QSFP ports of both machines and give both ends an address on the same subnet.",
					"ollama cluster init")
			}

			if m.container != nil {
				if err := dockerAvailable(ctx); err != nil {
					r.problem("Docker", "unavailable", err.Error(), "add your user to the docker group")
				} else {
					r.ok("Docker", "available")
				}
			}

			cfg := m.config()
			if len(cfg.Workers) > 0 {
				for _, w := range cfg.Workers {
					r.info("Worker", fmt.Sprintf("%s at %s (already added)", w.Label(), w.Endpoint()))
				}
			}

			if r.problems > 0 {
				return fmt.Errorf("%d problem(s) to fix first", r.problems)
			}
			fmt.Println("Ready. Next step: ollama cluster add <address of the second machine>")
			return nil
		},
	}
}

// --- add --------------------------------------------------------------------

type addOptions struct {
	addr          string
	image         string
	yes           bool
	allowLAN      bool
	keepNetwork   bool
	contextLength int
	rpcPort       int
	statusPort    int
	name          string
	mode          string
	cache         bool
	testModel     string
	noTest        bool

	// locationsOnly limits a container update to mounting the locations.
	locationsOnly bool
}

func addCmd() *cobra.Command {
	var o addOptions
	cmd := &cobra.Command{
		Use:   "add <user@host>",
		Short: "Attach a second machine: install the worker there and enable it here",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runAdd(cmd, args[0], o)
		},
	}
	f := cmd.Flags()
	f.StringVar(&o.addr, "addr", "", "worker address on the direct cable (detected when empty)")
	f.StringVar(&o.image, "image", "", "cluster image for Docker (default: the image of the running container)")
	f.BoolVarP(&o.yes, "yes", "y", false, "do not ask before restarting Ollama")
	f.BoolVar(&o.allowLAN, "allow-lan", false, "allow a worker on the local network instead of a direct cable (the RPC server has no authentication)")
	f.BoolVar(&o.keepNetwork, "keep-network", false, "keep the container network mode (RDMA may not work outside host networking)")
	f.IntVar(&o.contextLength, "context", defaultContextLength, "OLLAMA_CONTEXT_LENGTH set when the container has none (0 leaves it unset)")
	f.IntVar(&o.rpcPort, "rpc-port", cluster.DefaultRPCPort, "worker RPC port")
	f.IntVar(&o.statusPort, "status-port", cluster.DefaultStatusPort, "worker status port")
	f.StringVar(&o.name, "name", "", "worker name (default: its hostname)")
	f.StringVar(&o.mode, "mode", "", "worker installation: docker or native (default: docker when available there)")
	f.BoolVar(&o.cache, "cache", false, "enable the worker tensor cache")
	f.StringVar(&o.testModel, "test-model", "", "model for the final split test (default: the smallest local model)")
	f.BoolVar(&o.noTest, "no-test", false, "skip the final split test")
	return cmd
}

// workerFacts is what the worker check script reports.
type workerFacts struct {
	Hostname   string
	GPU        string
	Driver     string
	CUDA       string
	MemTotal   uint64
	MemAvail   uint64
	DiskFree   uint64
	Docker     string
	NVRuntime  bool
	Infiniband bool
	Worker     string // state and image of an existing worker container
	Interfaces []netIf
}

const workerFactsScript = `
echo "hostname=$(hostname)"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
echo "cuda=$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}')"
echo "memtotal=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)"
echo "memavail=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)"
echo "diskfree=$(df -Pk "$HOME" | awk 'NR==2{print $4}')"
echo "docker=$(docker info --format '{{.ServerVersion}}' 2>/dev/null)"
echo "nvruntime=$(docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -c nvidia)"
echo "infiniband=$(test -d /dev/infiniband && echo yes)"
echo "worker=$(docker inspect ` + workerContainer + ` --format '{{.State.Status}} {{.Image}}' 2>/dev/null)"
echo "--interfaces--"
` + remoteInterfacesScript

func collectWorkerFacts(ctx context.Context, rm remote) (*workerFacts, error) {
	out, err := rm.run(ctx, workerFactsScript)
	if err != nil {
		return nil, err
	}
	f := &workerFacts{}
	head, ifaces, _ := strings.Cut(out, "--interfaces--")
	sc := bufio.NewScanner(strings.NewReader(head))
	for sc.Scan() {
		k, v, ok := strings.Cut(sc.Text(), "=")
		if !ok {
			continue
		}
		v = strings.TrimSpace(v)
		switch k {
		case "hostname":
			f.Hostname = v
		case "gpu":
			f.GPU = v
		case "driver":
			f.Driver = v
		case "cuda":
			f.CUDA = v
		case "memtotal":
			n, _ := strconv.ParseUint(v, 10, 64)
			f.MemTotal = n << 10
		case "memavail":
			n, _ := strconv.ParseUint(v, 10, 64)
			f.MemAvail = n << 10
		case "diskfree":
			n, _ := strconv.ParseUint(v, 10, 64)
			f.DiskFree = n << 10
		case "docker":
			f.Docker = v
		case "nvruntime":
			f.NVRuntime = v != "" && v != "0"
		case "infiniband":
			f.Infiniband = v == "yes"
		case "worker":
			f.Worker = v
		}
	}
	f.Interfaces = parseRemoteInterfaces(ifaces)
	return f, nil
}

func runAdd(cmd *cobra.Command, target string, o addOptions) error {
	ctx := cmd.Context()
	start := time.Now()
	rm := remote{target: target}

	// 1. This machine.
	m, err := detectMain(ctx, cmd)
	if err != nil {
		return err
	}
	step("Ollama %s here (%s)", m.version, m.where())

	image := o.image
	if m.container != nil {
		if image == "" {
			image = m.container.Config.Image
		}
		if !imageHasCluster(ctx, image) {
			return fmt.Errorf("image %s has no cluster support (no RPC worker inside); %s", image, clusterImageHint)
		}
		step("Image %s has the RPC module and worker", image)
	} else if errors.Is(m.statusErr, errNoClusterAPI) {
		return fmt.Errorf("the running native Ollama has no cluster support; install an ollama-cluster build first")
	}

	// 2. The worker, checked before anything changes.
	step("Connecting to %s over SSH", target)
	if err := rm.ensureAccess(ctx); err != nil {
		return err
	}
	facts, err := collectWorkerFacts(ctx, rm)
	if err != nil {
		return fmt.Errorf("checking %s: %w", target, err)
	}
	name := o.name
	if name == "" {
		name = facts.Hostname
	}

	localName, _, localCUDA := localGPU(ctx)
	switch {
	case facts.GPU == "":
		return fmt.Errorf("%s: no NVIDIA GPU found (nvidia-smi does not answer); nothing was changed", name)
	case localName != "" && facts.GPU != localName:
		return fmt.Errorf("%s has %s, this machine has %s; both machines need the same GPU. Nothing was changed", name, facts.GPU, localName)
	case localCUDA != "" && facts.CUDA != "" && majorOf(facts.CUDA) != majorOf(localCUDA):
		return fmt.Errorf("%s supports CUDA %s, this machine CUDA %s; update the NVIDIA driver on the older one. Nothing was changed", name, facts.CUDA, localCUDA)
	}
	step("%s: %s, driver %s, CUDA %s, %s memory (%s free)", name, facts.GPU, facts.Driver, facts.CUDA, humanBytes(facts.MemTotal), humanBytes(facts.MemAvail))

	link, ok := findLink(localLinkInterfaces(), facts.Interfaces, o.addr, o.allowLAN)
	if !ok {
		if o.addr != "" {
			return fmt.Errorf("address %s is not on a direct cable to this machine (no RDMA port on the same subnet on both ends). Use --allow-lan only if you accept an unauthenticated RPC port on your network. Nothing was changed", o.addr)
		}
		return errors.New("no direct cable between the machines: no RDMA port on this machine shares a subnet with one on the worker. Connect the QSFP ports, give both ends an address (e.g. 10.100.48.1 and .2) and run 'ollama cluster add' again. Nothing was changed")
	}
	if err := ping(ctx, link.Remote.Addr.String()); err != nil {
		return fmt.Errorf("%s does not answer ping on the cable address %s; check the cable. Nothing was changed", name, link.Remote.Addr)
	}
	kind := "RDMA " + link.Local.RDMADev + " <-> " + link.Remote.RDMADev
	if link.Local.RDMADev == "" || link.Remote.RDMADev == "" {
		kind = "local network, no RDMA (--allow-lan)"
	}
	step("Cable: %s (%s) <-> %s (%s), %s", link.Local.Addr, link.Local.Name, link.Remote.Addr, link.Remote.Name, kind)

	mode := o.mode
	if mode == "" {
		// --gpus works through the NVIDIA container toolkit hook, which does
		// not always register a separate runtime, so Docker itself decides
		mode = "native"
		if facts.Docker != "" && m.container != nil {
			mode = "docker"
		}
	}
	if mode == "docker" && m.container == nil {
		return errors.New("--mode docker needs the main Ollama in Docker (the worker runs the same image); use --mode native")
	}

	w := cluster.Worker{
		Name:       name,
		SSH:        target,
		Addr:       link.Remote.Addr.String(),
		LocalAddr:  link.Local.Addr.String(),
		RPCPort:    o.rpcPort,
		StatusPort: o.statusPort,
		Mode:       mode,
		Cache:      o.cache,
	}
	if err := w.Validate(); err != nil {
		return err
	}

	// 3. Worker installation.
	switch mode {
	case "docker":
		if facts.Docker == "" {
			return fmt.Errorf("%s: Docker is not available to %s (add the user to the docker group). Nothing was changed", name, rm.user())
		}
		w.Image = image
		if err := installDockerWorker(ctx, rm, m, w, image, facts, o.cache); err != nil {
			return err
		}
	case "native":
		if err := installNativeWorker(ctx, rm, m, image, w, o.cache); err != nil {
			return err
		}
	default:
		return fmt.Errorf("unknown --mode %q (docker or native)", mode)
	}

	step("Waiting for the worker to report")
	check, err := waitWorker(ctx, w, 120*time.Second)
	if err != nil {
		return err
	}
	if check.Status != nil {
		var total uint64
		for _, d := range check.Status.Devices {
			total += d.TotalMemory
		}
		step("Worker up: llama.cpp %s, %s, transport %s", check.Status.LlamaCpp, humanBytes(total), check.Status.Transport)
	}

	// 4. Firewall: the RPC port only for this machine.
	ensureFirewall(ctx, rm, w)

	// 5. This Ollama.
	if m.container != nil {
		changed, err := updateMainContainer(ctx, m, image, o)
		if err != nil {
			return err
		}
		if changed {
			if err := m.api.waitReady(ctx, 120*time.Second); err != nil {
				return err
			}
			step("Ollama restarted")
		} else {
			step("Ollama container already set up, no restart needed")
		}
	} else {
		ensureNativeContext(ctx, o)
	}

	// 6. Register the worker.
	st, err := m.api.status(ctx)
	if err != nil {
		return fmt.Errorf("cluster status after the restart: %w", err)
	}
	cfg := st.Config
	cfg.Workers = slices.DeleteFunc(cfg.Workers, func(x cluster.Worker) bool { return x.Addr == w.Addr || x.SSH == w.SSH })
	cfg.Workers = append(cfg.Workers, w)
	if err := m.api.saveConfig(ctx, &cfg); err != nil {
		return fmt.Errorf("saving the cluster configuration: %w", err)
	}
	step("Worker %s registered in %s", w.Label(), st.ConfigPath)

	// 7. Final test.
	st, err = m.api.status(ctx)
	if err != nil {
		return err
	}
	var localMem, remoteMem uint64
	for _, d := range st.Devices {
		if d.Remote {
			remoteMem += d.TotalMemory
		} else {
			localMem += d.TotalMemory
		}
	}
	if remoteMem == 0 {
		for _, c := range st.Workers {
			if c.Problem != "" {
				return fmt.Errorf("Ollama does not see the worker: %s Run: %s", c.Problem, c.Fix)
			}
		}
		return errors.New("Ollama does not see the worker device; run 'ollama cluster doctor'")
	}
	step("Ollama sees %d devices: %s here + %s on %s", len(st.Devices), humanBytes(localMem), humanBytes(remoteMem), w.Label())

	if !o.noTest {
		res, err := splitTest(ctx, m.api, o.testModel)
		if err != nil {
			return fmt.Errorf("split test: %w", err)
		}
		step("Test: %s loaded in %s, %s here + %s on %s, answered %q",
			res.model, humanDuration(res.load), humanBytes(res.local), humanBytes(res.remote), w.Label(), res.answer)
	}

	fmt.Printf("\nDone in %s. Ollama can now use %s of GPU memory on two machines.\n", humanDuration(time.Since(start)), humanBytes(localMem+remoteMem))
	fmt.Println("Models that fit here stay here; larger ones are split automatically.")
	fmt.Println("Check:   ollama cluster status")
	fmt.Println("Undo:    ollama cluster remove")
	return nil
}

func majorOf(v string) string {
	major, _, _ := strings.Cut(v, ".")
	return major
}

func waitWorker(ctx context.Context, w cluster.Worker, timeout time.Duration) (cluster.WorkerCheck, error) {
	deadline := time.Now().Add(timeout)
	var last cluster.WorkerCheck
	for {
		st, err := cluster.ProbeWorker(ctx, w)
		if err == nil && st.Running && len(st.Devices) > 0 {
			return cluster.WorkerCheck{Worker: w, Status: st, Usable: true}, nil
		}
		last = cluster.WorkerCheck{Worker: w, Status: st}
		if err != nil {
			last.Problem = err.Error()
		} else if st.LastError != "" {
			last.Problem = st.LastError
		}
		if time.Now().After(deadline) {
			return last, fmt.Errorf("the worker did not start within %s: %s (see 'docker logs %s' on the worker)", timeout, last.Problem, workerContainer)
		}
		select {
		case <-ctx.Done():
			return last, ctx.Err()
		case <-time.After(2 * time.Second):
		}
	}
}

func workerArgs(w cluster.Worker, cache bool) []string {
	args := []string{"cluster", "worker", "--addr", w.Addr,
		"--port", strconv.Itoa(w.RPCPort), "--status-port", strconv.Itoa(w.StatusPort)}
	// upgrade passes false and keeps what add set
	if cache || w.Cache {
		args = append(args, "--cache")
	}
	return args
}

// installDockerWorker runs the worker from the same image as the main Ollama,
// which guarantees both sides come from one llama.cpp build.
func installDockerWorker(ctx context.Context, rm remote, m *mainOllama, w cluster.Worker, image string, facts *workerFacts, cache bool) error {
	localID, err := imageID(ctx, image)
	if err != nil {
		return err
	}

	remoteID, _ := rm.run(ctx, fmt.Sprintf("docker image inspect --format '{{.Id}}' %s 2>/dev/null", shellQuote(image)))
	if strings.TrimSpace(remoteID) != localID {
		size, _ := docker(ctx, "image", "inspect", "--format", "{{.Size}}", image)
		n, _ := strconv.ParseUint(size, 10, 64)
		step("Copying image %s to %s (%s, once per version)", image, w.Label(), humanBytes(n))
		t := time.Now()
		via := rm
		// the direct cable is much faster than the LAN when SSH answers there
		if u := rm.user(); u != "" {
			cable := remote{target: u + "@" + w.Addr}
			if _, err := cable.run(ctx, "true"); err == nil {
				via = cable
			}
		}
		if err := via.pipeTo(ctx, exec.CommandContext(ctx, "docker", "save", image), "docker load"); err != nil {
			return fmt.Errorf("copying the image: %w", err)
		}
		step("Image copied in %s", humanDuration(time.Since(t)))
	} else {
		step("Image already on %s", w.Label())
	}

	args := workerArgs(w, cache)
	want := strings.Join(args, " ")
	current, _ := rm.run(ctx, fmt.Sprintf("docker inspect %s --format '{{.Image}}|{{.State.Status}}|{{index .Config.Labels \"ollama.cluster.args\"}}' 2>/dev/null", workerContainer))
	if parts := strings.SplitN(strings.TrimSpace(current), "|", 3); len(parts) == 3 && parts[0] == localID && parts[1] == "running" && parts[2] == want {
		step("Worker container already running with these settings")
		return nil
	}

	run := []string{"docker", "run", "-d", "--name", workerContainer, "--restart", "unless-stopped",
		"--gpus", "all", "--network", "host", "--ulimit", "memlock=-1:-1",
		"--label", "ollama.cluster.args=" + want, "--entrypoint", "/usr/bin/ollama"}
	if facts.Infiniband {
		run = append(run, "--device", "/dev/infiniband")
	}
	run = append(run, image)
	run = append(run, args...)

	quoted := make([]string, len(run))
	for i, a := range run {
		quoted[i] = shellQuote(a)
	}
	// a native worker from an earlier 'add --mode native' holds the ports
	script := fmt.Sprintf("%s\ndocker rm -f %s >/dev/null 2>&1; %s", stopNativeWorker, workerContainer, strings.Join(quoted, " "))
	step("Starting the worker on %s (listens only on %s)", w.Label(), w.Addr)
	if _, err := rm.run(ctx, script); err != nil {
		return fmt.Errorf("starting the worker container: %w", err)
	}
	return nil
}

// installNativeWorker copies the Ollama binary and its libraries to the
// worker and runs `ollama cluster worker` as a systemd user service.
func installNativeWorker(ctx context.Context, rm remote, m *mainOllama, image string, w cluster.Worker, cache bool) error {
	dir := "$HOME/.ollama-cluster/worker"
	step("Copying Ollama with its llama.cpp libraries to %s:%s", w.Label(), "~/.ollama-cluster/worker")

	var src *exec.Cmd
	if m.container != nil {
		// from the cluster image, not the running container: during 'add' the
		// container may still run an image without the worker
		src = exec.CommandContext(ctx, "docker", "run", "--rm", "--entrypoint", "tar", image, "c", "-C", "/", "usr/bin/ollama", "usr/lib/ollama")
	} else {
		exe, err := os.Executable()
		if err != nil {
			return err
		}
		root := strings.TrimSuffix(exe, "/bin/ollama")
		if root == exe {
			return fmt.Errorf("cannot find the install root of %s (expected <root>/bin/ollama)", exe)
		}
		src = exec.CommandContext(ctx, "tar", "c", "-C", root, "bin/ollama", "lib/ollama")
	}
	if err := rm.pipeTo(ctx, src, "rm -rf "+dir+".new && mkdir -p "+dir+".new && tar x -C "+dir+".new && rm -rf "+dir+" && mv "+dir+".new "+dir); err != nil {
		return err
	}

	bin := dir + "/usr/bin/ollama"
	if m.container == nil {
		bin = dir + "/bin/ollama"
	}
	args := workerArgs(w, cache)
	unit := fmt.Sprintf(`[Unit]
Description=Ollama cluster worker
After=network-online.target

[Service]
ExecStart=%s %s
Restart=always
RestartSec=3
LimitMEMLOCK=infinity

[Install]
WantedBy=default.target
`, strings.ReplaceAll(bin, "$HOME", "%h"), strings.Join(args, " "))

	script := fmt.Sprintf(`docker rm -f `+workerContainer+` >/dev/null 2>&1
set -e
mkdir -p "$HOME/.config/systemd/user"
cat > "$HOME/.config/systemd/user/ollama-cluster-worker.service" <<'UNIT'
%sUNIT
systemctl --user daemon-reload
systemctl --user enable ollama-cluster-worker.service >/dev/null 2>&1
systemctl --user restart ollama-cluster-worker.service
loginctl show-user "$(id -un)" -p Linger 2>/dev/null
`, unit)
	out, err := rm.run(ctx, script)
	if err != nil {
		return fmt.Errorf("starting the worker service: %w", err)
	}
	if !strings.Contains(out, "Linger=yes") {
		fmt.Printf("  The worker runs as a user service. To start it at boot without logging in, run once on %s:\n    sudo loginctl enable-linger %s\n", w.Label(), rm.user())
		if interactive() && confirm("  Run it now (asks for the sudo password on the worker)?", true) {
			_ = rm.runTTY(ctx, "sudo loginctl enable-linger $(id -un)")
		}
	}
	return nil
}

// stopNativeWorker stops and removes the systemd user service of a native
// worker; it does nothing when there is none.
const stopNativeWorker = `if [ -f "$HOME/.config/systemd/user/ollama-cluster-worker.service" ]; then
  systemctl --user disable --now ollama-cluster-worker.service >/dev/null 2>&1
  rm -f "$HOME/.config/systemd/user/ollama-cluster-worker.service"
  systemctl --user daemon-reload
  rm -rf "$HOME/.ollama-cluster/worker"
fi`

func firewallRule(w cluster.Worker) string {
	return fmt.Sprintf("INPUT ! -i lo -p tcp -m multiport --dports %d,%d ! -s %s -j DROP", w.RPCPort, w.StatusPort, shellQuote(w.LocalAddr))
}

// firewallState reports "present", "absent" or "unknown" (sudo needs a password).
func firewallState(ctx context.Context, rm remote, w cluster.Worker) string {
	out, _ := rm.run(ctx, fmt.Sprintf("sudo -n iptables -C %s 2>&1; echo rc=$?", firewallRule(w)))
	switch {
	case strings.Contains(out, "rc=0"):
		return "present"
	case strings.Contains(out, "password") || strings.Contains(out, "sudo:"):
		return "unknown"
	default:
		return "absent"
	}
}

func ensureFirewall(ctx context.Context, rm remote, w cluster.Worker) {
	if w.LocalAddr == "" {
		return
	}
	switch firewallState(ctx, rm, w) {
	case "present":
		step("Firewall on %s already admits only %s to the worker ports", w.Label(), w.LocalAddr)
		return
	case "absent":
		if _, err := rm.run(ctx, "sudo -n iptables -I "+firewallRule(w)); err == nil {
			step("Firewall on %s: worker ports open only to %s", w.Label(), w.LocalAddr)
			return
		}
	}

	cmdline := "sudo iptables -I " + firewallRule(w)
	if interactive() {
		fmt.Printf("  The RPC server has no authentication. A firewall rule on %s will admit only %s.\n  This needs sudo on %s once (the rule does not survive a reboot; 'ollama cluster doctor' checks it).\n", w.Label(), w.LocalAddr, w.Label())
		if confirm("  Add the rule now?", true) {
			if err := rm.runTTY(ctx, cmdline); err == nil {
				step("Firewall on %s: worker ports open only to %s", w.Label(), w.LocalAddr)
				return
			}
		}
	}
	fmt.Printf("  WARNING: no firewall rule on %s. The worker listens only on the cable address %s; to also block other hosts run there:\n    %s\n", w.Label(), w.Addr, cmdline)
}

// updateMainContainer recreates the main container with what the cluster
// needs: the cluster image, host networking, the RDMA device, unlimited
// locked memory, a context limit and the model locations.
// hostListenAddr is the OLLAMA_HOST that keeps Ollama reachable exactly as
// its published port 11434 was once the container uses host networking: the
// same host port and the same bind address, so a port published only on
// 127.0.0.1 does not become reachable from the network. Empty when nothing
// needs setting (all interfaces, port 11434, or no published port).
func hostListenAddr(ports []string) string {
	var addrs []string
	for _, p := range ports {
		// [hostIP:]hostPort:containerPort[/proto]; hostIP may be IPv6
		parts := strings.Split(p, ":")
		if len(parts) < 2 || strings.Split(parts[len(parts)-1], "/")[0] != "11434" {
			continue
		}
		hostPort := parts[len(parts)-2]
		hostIP := strings.Trim(strings.Join(parts[:len(parts)-2], ":"), "[]")
		if hostIP == "" || hostIP == "0.0.0.0" || hostIP == "::" {
			if hostPort == "11434" {
				return ""
			}
			return "0.0.0.0:" + hostPort
		}
		addrs = append(addrs, net.JoinHostPort(hostIP, hostPort))
	}
	if len(addrs) == 0 {
		return ""
	}
	// Ollama listens on one address; prefer IPv4 loopback when published there
	slices.Sort(addrs)
	for _, a := range addrs {
		if strings.HasPrefix(a, "127.") {
			return a
		}
	}
	return addrs[0]
}

func updateMainContainer(ctx context.Context, m *mainOllama, image string, o addOptions) (bool, error) {
	c := m.container
	prev := specFromContainer(ctx, c)
	spec := specFromContainer(ctx, c)
	spec.Image = image

	var changes []string
	if image != c.Config.Image {
		changes = append(changes, fmt.Sprintf("image %s -> %s", c.Config.Image, image))
	} else if id, err := imageID(ctx, image); err == nil && id != c.Image && !o.locationsOnly {
		changes = append(changes, "image "+image+" was updated")
	}

	if !o.locationsOnly && !o.keepNetwork && spec.NetworkMode != "host" {
		spec.NetworkMode = "host"
		if listen := hostListenAddr(spec.Ports); listen != "" {
			spec.setEnv("OLLAMA_HOST", listen)
			changes = append(changes, "host networking, listening on "+listen+" as the published port did (needed for RDMA)")
		} else {
			changes = append(changes, "host networking (same port, needed for RDMA)")
		}
	}
	if _, err := os.Stat("/dev/infiniband"); err == nil && !o.locationsOnly && !c.hasDevice("/dev/infiniband") {
		spec.addDevice("/dev/infiniband")
		changes = append(changes, "RDMA device /dev/infiniband")
	}
	if !o.locationsOnly && !c.memlockUnlimited() {
		spec.setUlimit("memlock", "-1:-1")
		changes = append(changes, "unlimited locked memory (RDMA)")
	}
	if _, ok := spec.getEnv("OLLAMA_CONTEXT_LENGTH"); !ok && !o.locationsOnly && o.contextLength > 0 {
		spec.setEnv("OLLAMA_CONTEXT_LENGTH", strconv.Itoa(o.contextLength))
		changes = append(changes, fmt.Sprintf("OLLAMA_CONTEXT_LENGTH=%d (without it large models take the full trained context)", o.contextLength))
	}
	for _, loc := range m.config().Locations {
		if !c.mounted(loc.Path) {
			spec.addBind(loc.Path)
			changes = append(changes, "location "+loc.Path)
		}
	}

	if len(changes) == 0 && reflect.DeepEqual(spec, prev) {
		return false, nil
	}

	fmt.Printf("  The Ollama container %q will be recreated with:\n", c.name())
	for _, ch := range changes {
		fmt.Printf("    - %s\n", ch)
	}
	fmt.Println("  Ollama stops answering for a few seconds; applications using it will see that.")
	if !o.yes && !confirm("  Restart Ollama now?", false) {
		return false, errors.New("Ollama was not restarted; the worker is installed. Run 'ollama cluster add' again when it is a good moment")
	}

	backup, err := backupContainer(ctx, c.name())
	if err != nil {
		return false, fmt.Errorf("backing up the container settings: %w", err)
	}
	st, _ := loadState()
	if st.Backup == "" || st.Container != c.name() {
		// keep the first backup: it is the state before the cluster
		st.Container, st.Backup, st.OriginalImage = c.name(), backup, c.Config.Image
	}
	st.ChangedAt = time.Now()
	if err := st.save(); err != nil {
		return false, err
	}
	step("Container settings saved to %s", backup)

	if err := recreate(ctx, spec, prev, c.Image); err != nil {
		return false, err
	}
	return true, nil
}

// ensureNativeContext reminds about the context limit for a native install,
// where the service environment is outside this command's reach.
func ensureNativeContext(ctx context.Context, o addOptions) {
	if v := os.Getenv("OLLAMA_CONTEXT_LENGTH"); v != "" || o.contextLength == 0 {
		return
	}
	out, err := exec.CommandContext(ctx, "systemctl", "show", "ollama", "-p", "Environment").Output()
	if err == nil && strings.Contains(string(out), "OLLAMA_CONTEXT_LENGTH=") {
		return
	}
	fmt.Printf("  Set a context limit for the Ollama service, otherwise large models take their full trained context:\n    sudo systemctl edit ollama   # [Service] Environment=\"OLLAMA_CONTEXT_LENGTH=%d\"\n", o.contextLength)
}

type splitResult struct {
	model         string
	load          time.Duration
	local, remote uint64
	answer        string
}

// splitTest loads a small model forced across both machines and checks that
// part of it sits on the worker.
func splitTest(ctx context.Context, a *apiClient, model string) (*splitResult, error) {
	if model == "" {
		var err error
		model, err = smallestIdleModel(ctx, a)
		if err != nil {
			return nil, err
		}
	}
	step("Test: loading %s split across both machines", model)

	split := true
	var resp api.GenerateResponse
	req := map[string]any{
		"model":      model,
		"prompt":     "Reply with the single word OK.",
		"stream":     false,
		"think":      false,
		"keep_alive": "2m",
		// a small context keeps the test small on servers without a context
		// limit, where even an 8B model would take its full trained context
		"options": map[string]any{"cluster_split": split, "num_predict": 16, "num_ctx": 4096, "temperature": 0},
	}
	if err := a.do(ctx, http.MethodPost, "/api/generate", req, &resp); err != nil {
		return nil, err
	}
	res := &splitResult{model: model, load: resp.LoadDuration, answer: strings.TrimSpace(resp.Response)}

	st, err := a.status(ctx)
	if err != nil {
		return nil, err
	}
	for _, l := range st.Loaded {
		if !sameModel(l.Model, model) {
			continue
		}
		for _, d := range l.Devices {
			if d.Remote {
				res.remote += d.Size
			} else {
				res.local += d.Size
			}
		}
	}
	_ = a.do(ctx, http.MethodPost, "/api/generate", map[string]any{"model": model, "keep_alive": 0}, nil)
	if res.remote == 0 {
		return res, fmt.Errorf("%s loaded, but nothing was placed on the worker; check 'ollama cluster doctor' and the Ollama log", model)
	}
	return res, nil
}

func sameModel(a, b string) bool {
	norm := func(s string) string {
		if !strings.Contains(s, ":") {
			s += ":latest"
		}
		return s
	}
	return norm(a) == norm(b) || strings.HasSuffix(norm(a), "/"+norm(b))
}

// smallestIdleModel picks the smallest generative model that is not loaded,
// so the test does not reload a model other applications are using.
func smallestIdleModel(ctx context.Context, a *apiClient) (string, error) {
	var tags api.ListResponse
	if err := a.do(ctx, http.MethodGet, "/api/tags", nil, &tags); err != nil {
		return "", err
	}
	var ps api.ProcessResponse
	_ = a.do(ctx, http.MethodGet, "/api/ps", nil, &ps)
	loaded := map[string]bool{}
	for _, p := range ps.Models {
		loaded[p.Name] = true
	}

	models := slices.DeleteFunc(tags.Models, func(m api.ListModelResponse) bool {
		n := strings.ToLower(m.Name)
		return loaded[m.Name] || m.Size == 0 || m.RemoteModel != "" ||
			strings.Contains(n, "embed") || strings.Contains(n, "rerank") || strings.Contains(n, "whisper")
	})
	if len(models) == 0 {
		return "", errors.New("no idle local model for the test; pass --test-model or --no-test")
	}
	sort.Slice(models, func(i, j int) bool { return models[i].Size < models[j].Size })
	return models[0].Name, nil
}

func shellQuote(s string) string {
	if s != "" && !strings.ContainsAny(s, " \t\n'\"\\$`!*?[]{}()<>|&;#~") {
		return s
	}
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

// --- remove -----------------------------------------------------------------

func removeCmd() *cobra.Command {
	var yes bool
	cmd := &cobra.Command{
		Use:   "remove",
		Short: "Detach the second machine and restore Ollama as it was before 'add'",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			m, err := detectMain(ctx, cmd)
			if err != nil {
				return err
			}
			st, err := loadState()
			if err != nil {
				return err
			}
			cfg := m.config()

			fmt.Println("This will:")
			for _, w := range cfg.Workers {
				fmt.Printf("  - stop and remove the worker on %s and its firewall rule\n", w.Label())
			}
			if st.Backup != "" && m.container != nil {
				fmt.Printf("  - recreate the Ollama container %q from %s (image %s); Ollama stops for a few seconds\n", st.Container, st.Backup, st.OriginalImage)
			}
			if len(cfg.Workers) == 0 && st.Backup == "" {
				fmt.Println("  nothing: no worker is configured and no container backup exists")
				return nil
			}
			if !yes && !confirm("Continue?", false) {
				return errors.New("nothing was changed")
			}

			for _, w := range cfg.Workers {
				if w.SSH == "" {
					step("Worker %s has no SSH target; remove it there by hand", w.Label())
					continue
				}
				rm := remote{target: w.SSH}
				script := fmt.Sprintf("docker rm -f %s >/dev/null 2>&1\n%s\ntrue", workerContainer, stopNativeWorker)
				if _, err := rm.run(ctx, script); err != nil {
					fmt.Printf("  WARNING: could not remove the worker on %s: %v\n", w.Label(), err)
				} else {
					step("Worker on %s removed", w.Label())
				}
				if w.LocalAddr != "" {
					switch firewallState(ctx, rm, w) {
					case "present":
						if _, err := rm.run(ctx, "sudo -n iptables -D "+firewallRule(w)); err == nil {
							step("Firewall rule on %s removed", w.Label())
						}
					case "unknown":
						cmdline := "sudo iptables -D " + firewallRule(w)
						if interactive() && confirm(fmt.Sprintf("  Remove the firewall rule on %s (sudo password there)?", w.Label()), true) {
							_ = rm.runTTY(ctx, cmdline+" 2>/dev/null; true")
						} else {
							fmt.Printf("  Remove the firewall rule on %s by hand if present:\n    %s\n", w.Label(), cmdline)
						}
					}
				}
			}

			if m.status != nil && len(cfg.Workers) > 0 {
				cfg.Workers = nil
				if err := m.api.saveConfig(ctx, cfg); err != nil {
					fmt.Printf("  WARNING: could not update the cluster configuration: %v\n", err)
				} else {
					step("Workers removed from the cluster configuration")
				}
			}

			if st.Backup != "" && m.container != nil {
				orig, err := loadBackup(st.Backup)
				if err != nil {
					return err
				}
				cur, err := inspectContainer(ctx, st.Container)
				if err != nil {
					return err
				}
				spec := specFromContainer(ctx, orig)
				if err := recreate(ctx, spec, specFromContainer(ctx, cur), cur.Image); err != nil {
					return err
				}
				if err := m.api.waitReady(ctx, 120*time.Second); err != nil {
					return err
				}
				step("Ollama container restored (image %s)", orig.Config.Image)
				_ = os.Remove(stateDir() + "/state.json")
			}

			v, err := m.api.version(ctx)
			if err != nil {
				return err
			}
			if st2, err := m.api.status(ctx); err == nil {
				for _, d := range st2.Devices {
					if d.Remote {
						return fmt.Errorf("Ollama %s still sees worker device %s", v, d.Name)
					}
				}
			}
			fmt.Printf("Done. Ollama %s answers and uses this machine only.\n", v)
			return nil
		},
	}
	cmd.Flags().BoolVarP(&yes, "yes", "y", false, "do not ask for confirmation")
	return cmd
}

// --- upgrade ----------------------------------------------------------------

func upgradeCmd() *cobra.Command {
	var image string
	var yes bool
	cmd := &cobra.Command{
		Use:   "upgrade",
		Short: "Bring Ollama and the workers to one build after an update",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			m, err := detectMain(ctx, cmd)
			if err != nil {
				return err
			}

			if m.container != nil {
				img := image
				if img == "" {
					img = m.container.Config.Image
				}
				if !imageHasCluster(ctx, img) {
					return fmt.Errorf("Ollama runs %s, an image without cluster support (an update replaced the cluster build?). Run 'ollama cluster upgrade --image <cluster image>'; %s", img, clusterImageHint)
				}
				changed, err := updateMainContainer(ctx, m, img, addOptions{yes: yes, keepNetwork: true})
				if err != nil {
					return err
				}
				if changed {
					if err := m.api.waitReady(ctx, 120*time.Second); err != nil {
						return err
					}
					if m, err = detectMain(ctx, cmd); err != nil {
						return err
					}
					step("Ollama restarted on %s", img)
				} else {
					step("Ollama already runs the current build of %s", img)
				}
			}
			if m.status == nil {
				return m.statusErr
			}
			cfg := m.config()
			if len(cfg.Workers) == 0 {
				fmt.Println("Done. No worker is configured, so only Ollama was checked.")
				return nil
			}

			for i, w := range cfg.Workers {
				if w.SSH == "" {
					if c := cluster.CheckWorker(ctx, w); !c.Usable {
						return fmt.Errorf("worker %s has no SSH target and does not match: %s", w.Label(), c.Problem)
					}
					continue
				}
				rm := remote{target: w.SSH}
				switch {
				case w.Mode == "docker" && m.container != nil:
					facts, err := collectWorkerFacts(ctx, rm)
					if err != nil {
						return err
					}
					// installDockerWorker skips a worker already on this image
					cfg.Workers[i].Image = m.container.Config.Image
					if err := installDockerWorker(ctx, rm, m, w, m.container.Config.Image, facts, false); err != nil {
						return err
					}
				case cluster.CheckWorker(ctx, w).Usable:
					step("Worker %s already matches", w.Label())
					continue
				default:
					img := ""
					if m.container != nil {
						img = m.container.Config.Image
					}
					if err := installNativeWorker(ctx, rm, m, img, w, false); err != nil {
						return err
					}
				}
				if _, err := waitWorker(ctx, w, 120*time.Second); err != nil {
					return err
				}
				c := cluster.CheckWorker(ctx, w)
				if !c.Usable {
					return fmt.Errorf("worker %s still does not match: %s", w.Label(), c.Problem)
				}
				step("Worker %s on Ollama %s, llama.cpp %s", w.Label(), c.Status.Version, c.Status.LlamaCpp)
			}
			if err := m.api.saveConfig(ctx, cfg); err != nil {
				return err
			}
			fmt.Println("Done. Ollama and the workers run one build.")
			return nil
		},
	}
	cmd.Flags().StringVar(&image, "image", "", "switch Ollama to this cluster image")
	cmd.Flags().BoolVarP(&yes, "yes", "y", false, "do not ask before restarting Ollama")
	return cmd
}
