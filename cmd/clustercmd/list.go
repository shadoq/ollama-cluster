package clustercmd

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cluster"
)

// dataBlob is the size from which a blob counts as model data (weights or
// projector) rather than metadata (template, parameters, license).
const dataBlob = 16 << 20

// fitMargin is added to the file size when estimating whether a model fits:
// context, compute buffers and the runner itself.
const fitMargin = 0.10

// --- ls ---------------------------------------------------------------------

type modelRow struct {
	Name         string   `json:"name"`
	Size         int64    `json:"size"`
	Family       string   `json:"family,omitempty"`
	Parameters   string   `json:"parameters,omitempty"`
	Quantization string   `json:"quantization,omitempty"`
	Context      int      `json:"context,omitempty"`
	Capabilities []string `json:"capabilities,omitempty"`
	Loaded       bool     `json:"loaded"`

	// Fits is "here" (one machine), "split" (needs the worker) or "too large".
	Fits string `json:"fits"`

	// Files is "store" when the data is in the Ollama store, a location
	// path when it is linked to a repository copy, "mixed", or "unknown"
	// when the server is too old to tell.
	Files  string             `json:"files"`
	Repo   string             `json:"repo,omitempty"`
	Broken int                `json:"broken_links,omitempty"`
	Blobs  []cluster.BlobFile `json:"blobs,omitempty"`
}

func lsCmd() *cobra.Command {
	var showFiles, asJSON bool
	cmd := &cobra.Command{
		Use:     "ls [filter]",
		Aliases: []string{"list"},
		Short:   "List models with context, capabilities, where their files are and whether they need the second machine",
		Args:    cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			m, err := clusterMain(cmd)
			if err != nil {
				return err
			}
			filter := ""
			if len(args) == 1 {
				filter = strings.ToLower(args[0])
			}
			rows, err := listRows(ctx, m, filter)
			if err != nil {
				return err
			}
			if asJSON {
				if !showFiles {
					for i := range rows {
						rows[i].Blobs = nil
					}
				}
				return printJSON(rows)
			}
			printRows(m, rows, showFiles)
			return nil
		},
	}
	cmd.Flags().BoolVar(&showFiles, "files", false, "show every data file of each model and where it is")
	cmd.Flags().BoolVar(&asJSON, "json", false, "print JSON")
	return cmd
}

func listRows(ctx context.Context, m *mainOllama, filter string) ([]modelRow, error) {
	var tags api.ListResponse
	if err := m.api.do(ctx, http.MethodGet, "/api/tags", nil, &tags); err != nil {
		return nil, err
	}
	var files cluster.ModelsResponse
	haveFiles := true
	if err := m.api.do(ctx, http.MethodGet, "/api/cluster/models", nil, &files); err != nil {
		var ae *apiError
		if !errors.As(err, &ae) || ae.Status != http.StatusNotFound {
			return nil, err
		}
		// an older cluster build: no file information
		haveFiles = false
	}
	blobs := map[string][]cluster.BlobFile{}
	for _, f := range files.Models {
		blobs[f.Model] = f.Blobs
	}
	var ps api.ProcessResponse
	_ = m.api.do(ctx, http.MethodGet, "/api/ps", nil, &ps)

	local, remote := memoryTotals(m.status)
	cfg := m.config()

	rows := []modelRow{}
	for _, t := range tags.Models {
		if filter != "" && !strings.Contains(strings.ToLower(t.Name), filter) {
			continue
		}
		r := modelRow{
			Name:         t.Name,
			Size:         t.Size,
			Family:       t.Details.Family,
			Parameters:   t.Details.ParameterSize,
			Quantization: t.Details.QuantizationLevel,
			Files:        "store",
		}
		for _, c := range t.Capabilities {
			r.Capabilities = append(r.Capabilities, string(c))
		}
		for _, p := range ps.Models {
			if sameModel(p.Name, t.Name) {
				r.Loaded = true
			}
		}
		r.Fits = fits(t.Size, local, remote)
		if haveFiles {
			r.Blobs = dataBlobs(blobs[t.Name])
			r.Files, r.Repo, r.Broken = whereFiles(cfg, r.Blobs)
		} else {
			r.Files = "unknown"
		}
		rows = append(rows, r)
	}

	// /api/show reads the model metadata: trained context and, for older
	// servers, the capabilities. A few at a time keeps it quick.
	var wg sync.WaitGroup
	sem := make(chan struct{}, 6)
	for i := range rows {
		wg.Add(1)
		go func(r *modelRow) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			var show api.ShowResponse
			if err := m.api.do(ctx, http.MethodPost, "/api/show", api.ShowRequest{Model: r.Name}, &show); err != nil {
				return
			}
			r.Context = trainedContext(show.ModelInfo)
			if len(r.Capabilities) == 0 {
				for _, c := range show.Capabilities {
					r.Capabilities = append(r.Capabilities, string(c))
				}
			}
		}(&rows[i])
	}
	wg.Wait()

	sort.Slice(rows, func(i, j int) bool { return rows[i].Name < rows[j].Name })
	return rows, nil
}

// memoryTotals returns the memory of the devices on this machine and on the
// workers.
func memoryTotals(st *cluster.StatusResponse) (local, remote uint64) {
	for _, d := range st.Devices {
		if d.Remote {
			remote += d.TotalMemory
		} else {
			local += d.TotalMemory
		}
	}
	return local, remote
}

func fits(size int64, local, remote uint64) string {
	need := uint64(float64(size) * (1 + fitMargin))
	switch {
	case local == 0 || need <= local:
		return "here"
	case remote > 0 && need <= local+remote:
		return "split"
	default:
		return "too large"
	}
}

func dataBlobs(bs []cluster.BlobFile) []cluster.BlobFile {
	var out []cluster.BlobFile
	for _, b := range bs {
		if b.Size >= dataBlob || (b.Target != "" && !b.OK) {
			out = append(out, b)
		}
	}
	return out
}

// whereFiles summarises where the data of a model is: the Ollama store, one
// location (with the repository) or several places.
func whereFiles(cfg *cluster.Config, bs []cluster.BlobFile) (where, repo string, broken int) {
	places := map[string]bool{}
	for _, b := range bs {
		if b.Target != "" && !b.OK {
			broken++
		}
		place, r := placeOf(cfg, b.Target)
		places[place] = true
		if r != "" {
			repo = r
		}
	}
	switch len(places) {
	case 0:
		return "store", "", broken
	case 1:
		for p := range places {
			where = p
		}
		return where, repo, broken
	default:
		return "mixed", repo, broken
	}
}

// placeOf returns the location holding target and the repository of the copy
// ("owner/repo"), or "store" for a blob that is not a link.
func placeOf(cfg *cluster.Config, target string) (place, repo string) {
	if target == "" {
		return "store", ""
	}
	for _, l := range cfg.Locations {
		rel, err := filepath.Rel(l.Path, target)
		if err != nil || strings.HasPrefix(rel, "..") {
			continue
		}
		parts := strings.Split(filepath.ToSlash(rel), "/")
		if len(parts) >= 3 && parts[0] == cluster.HFDir {
			repo = parts[1] + "/" + parts[2]
		}
		return l.Path, repo
	}
	return filepath.Dir(target), ""
}

func trainedContext(info map[string]any) int {
	arch, _ := info["general.architecture"].(string)
	if v, ok := info[arch+".context_length"].(float64); ok {
		return int(v)
	}
	return 0
}

func printRows(m *mainOllama, rows []modelRow, showFiles bool) {
	if len(rows) == 0 {
		fmt.Println("No models.")
		return
	}
	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "NAME\tSIZE\tPARAMS\tQUANT\tCONTEXT\tCAPABILITIES\tRUNS\tFILES")
	for _, r := range rows {
		name := r.Name
		if r.Loaded {
			name += " *"
		}
		files := r.Files
		if files == "store" {
			files = "Ollama store"
		}
		if r.Repo != "" {
			files += " (hf " + r.Repo + ")"
		}
		if r.Broken > 0 {
			files += fmt.Sprintf(" BROKEN: %d file(s) unreachable", r.Broken)
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", name, humanBytes(uint64(r.Size)), dash(r.Parameters), dash(r.Quantization), contextLabel(r.Context), capabilities(r.Capabilities), r.Fits, files)
		if showFiles {
			for _, b := range r.Blobs {
				where := "Ollama store"
				if b.Target != "" {
					where = b.Target
					if !b.OK {
						where += "  UNREACHABLE"
					}
				}
				fmt.Fprintf(tw, "  %s\t%s\t%s\t\t\t\t\t\n", shortDigest(b.Digest), humanBytes(uint64(b.Size)), where)
			}
		}
	}
	tw.Flush()

	// disk use counts every blob once, also when models share it
	seen := map[string]bool{}
	var store, copies int64
	for _, r := range rows {
		for _, b := range r.Blobs {
			if seen[b.Digest] {
				continue
			}
			seen[b.Digest] = true
			if b.Target == "" {
				store += b.Size
			} else {
				copies += b.Size
			}
		}
	}
	local, remote := memoryTotals(m.status)
	fmt.Printf("\n%d model(s); data on disk: %s in the Ollama store, %s in repository copies.\n", len(rows), humanBytes(uint64(store)), humanBytes(uint64(copies)))
	fmt.Println("*  loaded now (details: ollama cluster ps)")
	runs := fmt.Sprintf("RUNS  estimate from the file size + %.0f%%: here = fits this machine (%s)", fitMargin*100, humanBytes(local))
	if remote > 0 {
		runs += fmt.Sprintf(", split = needs the worker too (%s together)", humanBytes(local+remote))
	} else {
		runs += ", split = needs a second machine (none attached)"
	}
	fmt.Println(runs)
	fmt.Println("FILES Ollama store = blobs in OLLAMA_MODELS; a path = repository copy in that location, linked into Ollama (ollama cluster models)")
	if len(rows) > 0 && rows[0].Files == "unknown" {
		fmt.Println("      unknown = the running Ollama is an older build that does not report files; update it: ollama cluster upgrade")
	}
}

func dash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

func contextLabel(n int) string {
	switch {
	case n <= 0:
		return "-"
	case n%1000 == 0:
		return fmt.Sprintf("%dk", n/1000)
	case n%1024 == 0:
		return fmt.Sprintf("%dk", n/1024)
	case n >= 1000:
		return fmt.Sprintf("%dk", n/1000)
	default:
		return fmt.Sprint(n)
	}
}

// capabilities drops "completion", which nearly every model has.
func capabilities(cs []string) string {
	var out []string
	for _, c := range cs {
		if c != "completion" {
			out = append(out, c)
		}
	}
	if len(out) == 0 {
		return "-"
	}
	sort.Strings(out)
	return strings.Join(out, ",")
}

func shortDigest(d string) string {
	d = strings.TrimPrefix(d, "sha256:")
	if len(d) > 12 {
		d = d[:12]
	}
	return d
}

func printJSON(v any) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}

// --- ps ---------------------------------------------------------------------

func psCmd() *cobra.Command {
	var asJSON bool
	cmd := &cobra.Command{
		Use:   "ps",
		Short: "Show loaded models: memory per machine, context, parallel slots, active requests",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			m, err := clusterMain(cmd)
			if err != nil {
				return err
			}
			if asJSON {
				return printJSON(m.status.Loaded)
			}
			printPS(m.status)
			return nil
		},
	}
	cmd.Flags().BoolVar(&asJSON, "json", false, "print JSON")
	return cmd
}

func printPS(st *cluster.StatusResponse) {
	loaded := append([]cluster.LoadedModel(nil), st.Loaded...)
	sort.Slice(loaded, func(i, j int) bool { return loaded[i].Model < loaded[j].Model })

	if len(loaded) == 0 {
		fmt.Println("No model is loaded.")
	} else {
		tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
		fmt.Fprintln(tw, "NAME\tMEMORY\tWHERE\tCONTEXT\tSLOTS\tACTIVE\tUNLOADS")
		for _, l := range loaded {
			size := l.Size
			if size == 0 {
				for _, d := range l.Devices {
					size += d.Size
				}
			}
			fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%d\t%s\n", l.Model, humanBytes(size), placement(st, l), loadedContext(l), dash(itoa(l.Parallel)), l.Active, unloads(l))
		}
		tw.Flush()
	}

	fmt.Println()
	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "MACHINE\tFREE\tTOTAL\tSTATE")
	for _, d := range st.Devices {
		if d.Remote {
			continue
		}
		fmt.Fprintf(tw, "this machine (%s)\t%s\t%s\tOllama %s\n", d.Name, humanBytes(d.FreeMemory), humanBytes(d.TotalMemory), st.Version)
	}
	for _, c := range st.Workers {
		// the worker's own report: the scheduler sees a busy worker as full
		var free, total uint64
		if c.Status != nil {
			for _, d := range c.Status.Devices {
				free += d.FreeMemory
				total += d.TotalMemory
			}
		}
		state := workerState(c)
		fmt.Fprintf(tw, "worker %s (%s)\t%s\t%s\t%s\n", c.Worker.Label(), c.Worker.Endpoint(), memOrDash(free, total), memOrDash(total, total), state)
	}
	tw.Flush()
	if len(st.Config.Workers) == 0 {
		fmt.Println("No second machine is attached; every model runs here (ollama cluster add <user@host>).")
	}
}

// placement says on which machines a loaded model sits and how much each holds.
func placement(st *cluster.StatusResponse, l cluster.LoadedModel) string {
	var local uint64
	var parts []string
	split := false
	for _, d := range l.Devices {
		if !d.Remote {
			local += d.Size
			continue
		}
		split = true
		name := d.Name
		for _, c := range st.Workers {
			if strings.Contains(d.Name, c.Worker.Endpoint()) {
				name = c.Worker.Label()
			}
		}
		parts = append(parts, fmt.Sprintf("%s %s", name, humanBytes(d.Size)))
	}
	if !split {
		if len(l.Devices) == 0 {
			return "here (CPU)"
		}
		return "here"
	}
	if local == 0 {
		return "worker: " + strings.Join(parts, " + ")
	}
	return "split: here " + humanBytes(local) + " + " + strings.Join(parts, " + ")
}

func loadedContext(l cluster.LoadedModel) string {
	if l.TrainContext > 0 && l.TrainContext != l.Context {
		return fmt.Sprintf("%s of %s", contextLabel(l.Context), contextLabel(l.TrainContext))
	}
	return contextLabel(l.Context)
}

func unloads(l cluster.LoadedModel) string {
	switch {
	case l.Loading:
		return "loading"
	case l.Active > 0:
		return "in use"
	case l.KeepAlive < 0:
		return "never"
	case l.ExpiresAt.IsZero():
		return "-"
	}
	d := time.Until(l.ExpiresAt)
	if d <= 0 {
		return "now"
	}
	return "in " + humanDuration(d)
}

func workerState(c cluster.WorkerCheck) string {
	if !c.Usable {
		return "NOT USED: " + c.Problem
	}
	state := "idle"
	if c.Busy {
		state = "holds a split model"
	}
	if c.Status != nil && c.Status.Transport != "" {
		state += ", " + c.Status.Transport
	}
	return state
}

func memOrDash(v, total uint64) string {
	if total == 0 {
		return "-"
	}
	return humanBytes(v)
}

func itoa(n int) string {
	if n == 0 {
		return ""
	}
	return fmt.Sprint(n)
}
