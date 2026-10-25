// Package clustercmd implements `ollama cluster`: commands that attach a
// second machine to an existing Ollama and keep models as Hugging Face
// repository copies in ordered locations.
//
// The commands run on the host of the main Ollama. They talk to the Ollama
// API (also when Ollama runs in Docker), to Docker and, over SSH, to the
// worker machine. Every command checks first and changes second, can be
// repeated safely, and explains each problem with one sentence and one fix.
package clustercmd

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"golang.org/x/term"

	"github.com/ollama/ollama/cluster"
	"github.com/ollama/ollama/cluster/worker"
	"github.com/ollama/ollama/envconfig"
)

// Command returns the `ollama cluster` command tree.
func Command() *cobra.Command {
	root := &cobra.Command{
		Use:     "cluster",
		Aliases: []string{"cl"},
		Short:   "Use the memory of a second machine and manage model copies",
		Long: `Attach a second machine (e.g. a second DGX Spark on a direct cable) to this
Ollama. The second machine runs only a small worker; this Ollama keeps one
address, one model list and the same API, and splits models that do not fit
here across both machines.

Models from Hugging Face can be kept as plain repository copies in several
locations, ordered from the fastest; Ollama links to the best copy.

Start with 'ollama cluster init' (checks only), then
'ollama cluster add <user@host>'. Day to day: 'ollama cluster status',
'ollama cluster ls', 'ollama cluster ps' and 'ollama cluster doctor'.
Where models go: 'ollama cluster policy'. 'ollama cl' is short for
'ollama cluster'.
Every command explains itself: 'ollama cluster <command> help' or
'ollama cluster help <command>'.`,
	}

	root.PersistentFlags().String("container", "", "Docker container of the main Ollama (detected when empty)")

	root.AddCommand(
		initCmd(),
		addCmd(),
		statusCmd(),
		lsCmd(),
		psCmd(),
		policyCmd(),
		doctorCmd(),
		upgradeCmd(),
		benchCmd(),
		removeCmd(),
		locationsCmd(),
		pullCmd(),
		modelsCmd(),
		copyCmd(),
		dropCmd(),
		mvCmd(),
		importCmd(),
		workerCmd(),
	)
	applyHelp(root)
	return root
}

func workerCmd() *cobra.Command {
	var opts worker.Options
	cmd := &cobra.Command{
		Use:   "worker",
		Short: "Run the worker on the second machine (started by 'ollama cluster add')",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return worker.Run(cmd.Context(), opts)
		},
	}
	cmd.Flags().StringVar(&opts.Addr, "addr", "", "direct-link address to listen on")
	cmd.Flags().IntVar(&opts.RPCPort, "port", cluster.DefaultRPCPort, "RPC port")
	cmd.Flags().IntVar(&opts.StatusPort, "status-port", cluster.DefaultStatusPort, "status port")
	cmd.Flags().BoolVar(&opts.Cache, "cache", false, "enable the RPC tensor cache")
	cmd.Flags().IntVar(&opts.Threads, "threads", 0, "CPU threads of the RPC server")
	cmd.Flags().IntVar(&opts.MaxSlots, "max-slots", cluster.DefaultMaxSlots, "RPC servers at most, one model each")
	cmd.Flags().DurationVar(&opts.SlotIdle, "slot-idle", cluster.DefaultSlotIdle, "how long a released extra RPC server is kept")
	return cmd
}

// --- Ollama API -----------------------------------------------------------

type apiClient struct {
	base string
	http *http.Client
}

func newAPI() *apiClient {
	return &apiClient{
		base: envconfig.Host().String(),
		http: &http.Client{Timeout: 0},
	}
}

type apiError struct {
	Status int
	Msg    string
}

func (e *apiError) Error() string { return e.Msg }

func (a *apiClient) do(ctx context.Context, method, path string, body, out any) error {
	var r io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return err
		}
		r = bytes.NewReader(data)
	}
	req, err := http.NewRequestWithContext(ctx, method, a.base+path, r)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := a.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode >= 300 {
		var e struct {
			Error string `json:"error"`
		}
		msg := strings.TrimSpace(string(data))
		if json.Unmarshal(data, &e) == nil && e.Error != "" {
			msg = e.Error
		}
		return &apiError{Status: resp.StatusCode, Msg: msg}
	}
	if out != nil {
		return json.Unmarshal(data, out)
	}
	return nil
}

// stream sends a request whose answer is one JSON object per line and calls
// fn for each; a line with an "error" field ends it with that error.
func (a *apiClient) stream(ctx context.Context, method, path string, body any, fn func(json.RawMessage) error) error {
	data, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, method, a.base+path, bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := a.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64<<10), 4<<20)
	for sc.Scan() {
		line := sc.Bytes()
		var e struct {
			Error string `json:"error"`
		}
		if json.Unmarshal(line, &e) == nil && e.Error != "" {
			return &apiError{Status: resp.StatusCode, Msg: e.Error}
		}
		if resp.StatusCode >= 300 {
			return &apiError{Status: resp.StatusCode, Msg: strings.TrimSpace(string(line))}
		}
		if err := fn(json.RawMessage(line)); err != nil {
			return err
		}
	}
	if err := sc.Err(); err != nil {
		return err
	}
	if resp.StatusCode >= 300 {
		return &apiError{Status: resp.StatusCode, Msg: resp.Status}
	}
	return nil
}

// version returns the server version, or an error when Ollama is down.
func (a *apiClient) version(ctx context.Context) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	var v struct {
		Version string `json:"version"`
	}
	err := a.do(ctx, http.MethodGet, "/api/version", nil, &v)
	return v.Version, err
}

// status returns the cluster status. errNoClusterAPI means the running Ollama
// is a build without cluster support.
func (a *apiClient) status(ctx context.Context) (*cluster.StatusResponse, error) {
	var st cluster.StatusResponse
	err := a.do(ctx, http.MethodGet, "/api/cluster/status", nil, &st)
	var ae *apiError
	if errors.As(err, &ae) && ae.Status == http.StatusNotFound {
		return nil, errNoClusterAPI
	}
	if err != nil {
		return nil, err
	}
	return &st, nil
}

var errNoClusterAPI = errors.New("the running Ollama has no cluster support (it is not an ollama-cluster build)")

func (a *apiClient) saveConfig(ctx context.Context, cfg *cluster.Config) error {
	return a.do(ctx, http.MethodPut, "/api/cluster/config", cfg, nil)
}

// waitReady waits until the Ollama API answers.
func (a *apiClient) waitReady(ctx context.Context, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for {
		if _, err := a.version(ctx); err == nil {
			return nil
		} else if time.Now().After(deadline) {
			return fmt.Errorf("Ollama did not come back within %s: %w", timeout, err)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Second):
		}
	}
}

// --- output -----------------------------------------------------------------

// report prints aligned check rows: name, value, OK or PROBLEM, and for a
// problem one sentence and the fix.
type report struct {
	problems int
	warnings int
}

func (r *report) ok(name, value string) {
	fmt.Printf("%-14s %-52s OK\n", name, value)
}

func (r *report) warn(name, value, why, fix string) {
	r.warnings++
	fmt.Printf("%-14s %-52s WARNING\n", name, value)
	r.explain(why, fix)
}

func (r *report) problem(name, value, why, fix string) {
	r.problems++
	fmt.Printf("%-14s %-52s PROBLEM\n", name, value)
	r.explain(why, fix)
}

func (r *report) info(name, value string) {
	fmt.Printf("%-14s %s\n", name, value)
}

func (r *report) explain(why, fix string) {
	if why != "" {
		fmt.Printf("%14s %s\n", "", why)
	}
	if fix != "" {
		fmt.Printf("%14s Fix: %s\n", "", fix)
	}
}

func step(format string, args ...any) {
	fmt.Printf("- "+format+"\n", args...)
}

// confirm asks a yes/no question. Without a terminal it returns def.
func confirm(question string, def bool) bool {
	if !term.IsTerminal(int(os.Stdin.Fd())) {
		return def
	}
	suffix := " [y/N] "
	if def {
		suffix = " [Y/n] "
	}
	fmt.Print(question + suffix)
	line, _ := bufio.NewReader(os.Stdin).ReadString('\n')
	line = strings.ToLower(strings.TrimSpace(line))
	if line == "" {
		return def
	}
	return line == "y" || line == "yes" || line == "t" || line == "tak"
}

func interactive() bool {
	return term.IsTerminal(int(os.Stdin.Fd()))
}

func humanBytes(b uint64) string {
	const unit = 1000
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := uint64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(b)/float64(div), "kMGTPE"[exp])
}

func humanDuration(d time.Duration) string {
	switch {
	case d < time.Minute:
		return fmt.Sprintf("%.0f s", d.Seconds())
	case d < time.Hour:
		return fmt.Sprintf("%d min %d s", int(d.Minutes()), int(d.Seconds())%60)
	default:
		return fmt.Sprintf("%d h %d min", int(d.Hours()), int(d.Minutes())%60)
	}
}
