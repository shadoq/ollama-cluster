package clustercmd

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/cluster"
)

// --- status -----------------------------------------------------------------

func statusCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "status",
		Short: "Show the machines, their memory and where loaded models sit",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			m, err := detectMain(ctx, cmd)
			if err != nil {
				return err
			}
			if m.status == nil {
				return m.statusErr
			}
			printStatus(m)
			return nil
		},
	}
}

func printStatus(m *mainOllama) {
	st := m.status
	fmt.Printf("Ollama    %s (%s), llama.cpp %s\n", st.Version, m.where(), st.LlamaCpp)
	fmt.Printf("Policy    %s\n", policySummary(&st.Config))

	fmt.Println("\nMachines")
	for _, d := range st.Devices {
		where := "this machine"
		if d.Remote {
			where = d.Description
		} else if d.Description != "" {
			where = d.Description + ", this machine"
		}
		fmt.Printf("  %-26s %-40s %9s total %9s free\n", d.Name, where, humanBytes(d.TotalMemory), humanBytes(d.FreeMemory))
	}
	for _, c := range st.Workers {
		switch {
		case !c.Usable:
			fmt.Printf("  %-26s PROBLEM: %s\n  %-26s Fix: %s\n", c.Worker.Label(), c.Problem, "", c.Fix)
		default:
			state := "idle"
			if c.Busy {
				state = "busy (a model is loaded on it)"
			}
			fmt.Printf("  worker %-19s %s, %s, %s\n", c.Worker.Label(), c.Worker.Endpoint(), c.Status.Transport, state)
		}
	}
	if len(st.Config.Workers) == 0 {
		fmt.Println("  no second machine; add one with: ollama cluster add <user@host>")
	}

	fmt.Println("\nLoaded models")
	if len(st.Loaded) == 0 {
		fmt.Println("  none")
	}
	for _, l := range st.Loaded {
		var parts []string
		for _, d := range l.Devices {
			parts = append(parts, fmt.Sprintf("%s on %s", humanBytes(d.Size), d.Name))
		}
		fmt.Printf("  %-34s context %-7d %s\n", l.Model, l.Context, strings.Join(parts, " + "))
	}

	if len(st.Config.Locations) > 0 {
		fmt.Println("\nModel locations (in order)")
		for i, l := range st.Config.Locations {
			fmt.Printf("  %d. %s%s\n", i+1, l.Path, locationNote(l))
		}
	}
}

func locationNote(l cluster.Location) string {
	var notes []string
	if l.Archive {
		notes = append(notes, "archive, never loaded from")
	}
	if free, ok := diskFree(l.Path); ok {
		notes = append(notes, humanBytes(free)+" free")
	} else {
		notes = append(notes, "NOT AVAILABLE")
	}
	return " (" + strings.Join(notes, ", ") + ")"
}

// --- doctor -----------------------------------------------------------------

func doctorCmd() *cobra.Command {
	var fix bool
	cmd := &cobra.Command{
		Use:   "doctor",
		Short: "Check everything the cluster depends on and explain each problem",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return runDoctor(cmd, fix)
		},
	}
	cmd.Flags().BoolVar(&fix, "fix", false, "repair what is safe to repair (model links)")
	return cmd
}

func runDoctor(cmd *cobra.Command, fix bool) error {
	ctx := cmd.Context()
	r := &report{}

	m, err := detectMain(ctx, cmd)
	if err != nil {
		r.problem("Ollama", "does not answer", err.Error(), "start Ollama")
		return errors.New("1 problem")
	}
	r.ok("Ollama", fmt.Sprintf("%s (%s)", m.version, m.where()))

	if m.status == nil {
		if errors.Is(m.statusErr, errNoClusterAPI) {
			r.problem("Cluster build", "no", "The running Ollama is not an ollama-cluster build (an update may have replaced it).", "ollama cluster upgrade --image <cluster image>")
		} else {
			r.problem("Cluster", "status unavailable", m.statusErr.Error(), "check the Ollama log")
		}
		return fmt.Errorf("%d problem(s)", r.problems)
	}
	st := m.status
	r.ok("llama.cpp", st.LlamaCpp)

	if c := m.container; c != nil {
		doctorContainer(ctx, r, m)
	}

	for _, loc := range st.Config.Locations {
		if _, ok := diskFree(loc.Path); !ok {
			r.problem("Location", loc.Path, "The directory is not available (disk not mounted or node powered off). Models linked there load from the next location that has a copy.", "mount it, or: ollama cluster locations remove "+loc.Path)
		} else if m.container != nil && !m.container.mounted(loc.Path) {
			r.problem("Location", loc.Path, "Ollama in Docker does not see this directory at the same path.", "ollama cluster add (it recreates the container with the location mounted)")
		} else {
			r.ok("Location", loc.Path+locationNote(loc))
		}
	}

	var links cluster.LinksResponse
	if err := m.api.do(ctx, http.MethodGet, "/api/cluster/links", nil, &links); err == nil {
		broken := 0
		for _, l := range links.Links {
			if !l.OK {
				broken++
			}
		}
		switch {
		case fix:
			var res cluster.RelinkResponse
			if err := m.api.do(ctx, http.MethodPost, "/api/cluster/relink", nil, &res); err != nil {
				r.problem("Model links", fmt.Sprintf("%d broken", broken), err.Error(), "")
			} else {
				missing := 0
				for _, c := range res.Changes {
					if c.Missing {
						missing++
						fmt.Printf("%14s missing everywhere: %s\n", "", c.From)
					}
				}
				if missing > 0 {
					r.problem("Model links", fmt.Sprintf("%d with no copy left", missing), "No location has these files anymore.", "ollama cluster pull <repo:tag> (downloads them again)")
				} else {
					r.ok("Model links", fmt.Sprintf("%d switched to the best copy, %d in total", len(res.Changes), len(links.Links)))
				}
			}
		case broken > 0:
			r.problem("Model links", fmt.Sprintf("%d of %d broken", broken, len(links.Links)), "Some model files point at a location that is not available.", "ollama cluster doctor --fix")
		case len(links.Links) > 0:
			r.ok("Model links", fmt.Sprintf("%d, all reachable", len(links.Links)))
		}
	}

	if len(st.Config.Workers) == 0 {
		r.info("Worker", "none configured (ollama cluster add <user@host>)")
	}
	for _, c := range st.Workers {
		doctorWorker(ctx, r, m, c)
	}
	for _, w := range st.Config.Workers {
		if w.Disabled {
			r.info("Worker", w.Label()+" disabled in the configuration")
		}
	}

	if m.container != nil {
		logs := containerLogs(ctx, m.container.name(), "24h")
		skipped := strings.Count(logs, "cluster worker skipped") + strings.Count(logs, "cluster worker unavailable at model start")
		if skipped > 0 {
			r.warn("Fallbacks", fmt.Sprintf("%d in the last 24 h", skipped), "Models loaded on this machine only because a worker was not usable. Ollama kept working.", "docker logs "+m.container.name()+" 2>&1 | grep 'cluster worker'")
		} else {
			r.ok("Fallbacks", "none in the last 24 h")
		}
	}

	if r.problems > 0 {
		return fmt.Errorf("%d problem(s)", r.problems)
	}
	fmt.Println("Everything checked.")
	return nil
}

// containerLogs returns the log of a container; Ollama logs to stderr.
func containerLogs(ctx context.Context, name, since string) string {
	out, _ := exec.CommandContext(ctx, "docker", "logs", "--since", since, name).CombinedOutput()
	return string(out)
}

func doctorContainer(ctx context.Context, r *report, m *mainOllama) {
	c := m.container
	if imageHasCluster(ctx, c.Config.Image) {
		r.ok("Image", c.Config.Image)
	} else {
		r.problem("Image", c.Config.Image, "The image has no RPC module.", "ollama cluster upgrade --image <cluster image>")
	}
	if len(m.status.Config.Workers) == 0 {
		return
	}
	if c.HostConfig.NetworkMode == "host" {
		r.ok("Network", "host")
	} else {
		r.warn("Network", c.HostConfig.NetworkMode, "RDMA needs host networking; the worker is reached over TCP, which is slower.", "ollama cluster add <worker> (recreates the container)")
	}
	if _, err := os.Stat("/dev/infiniband"); err == nil {
		if c.hasDevice("/dev/infiniband") {
			r.ok("RDMA device", "/dev/infiniband in the container")
		} else {
			r.warn("RDMA device", "missing in the container", "Without it the link falls back to TCP.", "ollama cluster add <worker>")
		}
	}
	if c.memlockUnlimited() {
		r.ok("Locked memory", "unlimited")
	} else {
		r.warn("Locked memory", "limited", "RDMA needs unlimited locked memory (--ulimit memlock=-1).", "ollama cluster add <worker>")
	}
	if v, ok := c.env("OLLAMA_CONTEXT_LENGTH"); ok {
		r.ok("Context limit", v)
	} else {
		r.warn("Context limit", "not set", "Large models take their full trained context and can fill both machines.", "ollama cluster add <worker> (sets OLLAMA_CONTEXT_LENGTH)")
	}
}

func doctorWorker(ctx context.Context, r *report, m *mainOllama, c cluster.WorkerCheck) {
	w := c.Worker
	name := "Worker"
	if !c.Usable {
		r.problem(name, w.Label(), c.Problem, c.Fix)
	} else {
		state := "idle"
		if c.Busy {
			state = "busy"
		}
		r.ok(name, fmt.Sprintf("%s %s, Ollama %s, %s", w.Label(), w.Endpoint(), c.Status.Version, state))
	}
	if c.Status != nil {
		switch {
		case strings.Contains(strings.ToUpper(c.Status.Transport), "RDMA"):
			r.ok("Transport", c.Status.Transport)
		case c.Status.Transport != "":
			r.warn("Transport", c.Status.Transport, "The link runs without RDMA, model loading and generation are slower.", "check /dev/infiniband and memlock on the worker; ollama cluster upgrade")
		}
		if c.Status.Restarts > 0 {
			r.warn("RPC restarts", strconv.Itoa(c.Status.Restarts), "The RPC server on the worker exited and was started again. Last error: "+c.Status.LastError, "ollama cluster status")
		}
	}

	if w.SSH == "" {
		return
	}
	rm := remote{target: w.SSH}
	out, err := rm.run(ctx, fmt.Sprintf(`docker inspect %s --format 'container={{.State.Status}} restarts={{.RestartCount}}' 2>/dev/null
echo "gpu_apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | grep -v -e ggml-rpc -e rpc-server | tr '\n' ';')"
echo "containers=$(docker ps --format '{{.Names}} {{.Image}}' 2>/dev/null | grep -i -e llama -e vllm -e sglang -e ollama | grep -v %s | tr '\n' ';')"
`, workerContainer, workerContainer))
	if err != nil {
		r.problem("SSH", w.SSH, "Cannot reach the worker over SSH: "+err.Error(), "ssh "+w.SSH)
		return
	}
	for _, line := range strings.Split(out, "\n") {
		switch {
		case strings.HasPrefix(line, "container="):
			f := strings.Fields(line)
			if len(f) == 2 && !strings.HasSuffix(f[1], "=0") {
				r.warn("Worker restarts", strings.TrimPrefix(f[1], "restarts="), "Docker restarted the worker container.", "ssh "+w.SSH+" docker logs "+workerContainer)
			}
		case strings.HasPrefix(line, "gpu_apps="):
			if v := strings.Trim(strings.TrimPrefix(line, "gpu_apps="), ";"); v != "" {
				r.warn("Other GPU use", v, "Other processes use the GPU of the worker, it has less memory for Ollama.", "")
			}
		case strings.HasPrefix(line, "containers="):
			if v := strings.Trim(strings.TrimPrefix(line, "containers="), ";"); v != "" {
				r.warn("Other servers", v, "Other model servers run on the worker and compete for its memory.", "")
			}
		}
	}
	if w.LocalAddr != "" {
		switch firewallState(ctx, rm, w) {
		case "present":
			r.ok("Firewall", "worker ports only for "+w.LocalAddr)
		case "absent":
			r.warn("Firewall", "no rule", fmt.Sprintf("The worker listens only on the cable address %s, but any host on that link can use it (no authentication).", w.Addr), "ssh -t "+w.SSH+" sudo iptables -I "+firewallRule(w))
		default:
			r.info("Firewall", "not checked (sudo on the worker asks for a password)")
		}
	}
}
