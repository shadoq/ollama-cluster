package clustercmd

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cluster"
)

// Hugging Face repository copies live in <location>/hf/<owner>/<repo>. The
// files keep their repository paths, so llama.cpp, vLLM or llama-swap can use
// the same copy, and copyManifest records what Ollama made of them.

const (
	copyManifestName = ".ollama-cluster.json"
	pullReserve      = 20 << 30 // free space left on a location after a pull
)

type copyFile struct {
	Path   string `json:"path"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256,omitempty"`
}

type copyManifest struct {
	Repo     string              `json:"repo"`
	Revision string              `json:"revision,omitempty"`
	Files    []copyFile          `json:"files"`
	Models   map[string][]string `json:"models,omitempty"` // Ollama name -> files
	Updated  time.Time           `json:"updated"`
}

// validRepo matches owner/repo names of Hugging Face.
var validRepo = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$`)

func checkRepo(repo string) error {
	if !validRepo.MatchString(repo) || strings.Contains(repo, "..") {
		return fmt.Errorf("%q is not a Hugging Face repository name (owner/repo)", repo)
	}
	return nil
}

// safeRelPath reports whether p, a path from Hugging Face or from a copy
// manifest, stays inside the copy directory.
func safeRelPath(p string) bool {
	if p == "" || strings.Contains(p, "\\") || path.IsAbs(p) || path.Clean(p) != p {
		return false
	}
	return !slices.Contains(strings.Split(p, "/"), "..")
}

func copyDir(loc, repo string) string {
	return filepath.Join(loc, cluster.HFDir, filepath.FromSlash(repo))
}

func readCopyManifest(dir string) (*copyManifest, error) {
	data, err := os.ReadFile(filepath.Join(dir, copyManifestName))
	if err != nil {
		return nil, err
	}
	var m copyManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	m.Files = slices.DeleteFunc(m.Files, func(f copyFile) bool { return !safeRelPath(f.Path) })
	return &m, nil
}

func (m *copyManifest) addFiles(files ...copyFile) {
	for _, f := range files {
		i := slices.IndexFunc(m.Files, func(x copyFile) bool { return x.Path == f.Path })
		if i < 0 {
			m.Files = append(m.Files, f)
		} else if f.SHA256 != "" || m.Files[i].Size != f.Size {
			m.Files[i] = f
		}
	}
	sort.Slice(m.Files, func(i, j int) bool { return m.Files[i].Path < m.Files[j].Path })
}

func (m *copyManifest) save(dir string) error {
	m.Updated = time.Now().UTC()
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	tmp := filepath.Join(dir, copyManifestName+".tmp")
	if err := os.WriteFile(tmp, append(data, '\n'), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, filepath.Join(dir, copyManifestName))
}

// mergeCopyManifest updates (or creates) the manifest of a copy.
func mergeCopyManifest(dir, repo, revision string, files []copyFile, model string, modelFiles []string) error {
	m, err := readCopyManifest(dir)
	if err != nil {
		m = &copyManifest{Repo: repo}
	}
	if revision != "" {
		m.Revision = revision
	}
	m.addFiles(files...)
	if model != "" {
		if m.Models == nil {
			m.Models = map[string][]string{}
		}
		m.Models[model] = modelFiles
	}
	return m.save(dir)
}

// copyComplete reports whether dir holds every file with the expected size.
func copyComplete(dir string, files []copyFile) bool {
	for _, f := range files {
		fi, err := os.Stat(filepath.Join(dir, filepath.FromSlash(f.Path)))
		if err != nil || !fi.Mode().IsRegular() || (f.Size > 0 && fi.Size() != f.Size) {
			return false
		}
	}
	return true
}

// copyVerified reports whether dir holds every file with the expected size
// and, where the listing has one, the expected sha256. Digests are kept per
// location, so an unchanged file is hashed once.
func copyVerified(cfg *cluster.Config, dir string, files []copyFile) bool {
	if !copyComplete(dir, files) {
		return false
	}
	for _, f := range files {
		if f.SHA256 != "" && !cluster.HasDigest(cfg, filepath.Join(dir, filepath.FromSlash(f.Path)), f.SHA256) {
			return false
		}
	}
	return true
}

func filesSize(files []copyFile) int64 {
	var n int64
	for _, f := range files {
		n += f.Size
	}
	return n
}

// --- locations --------------------------------------------------------------

func locationsCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "locations",
		Short: "List and order the directories that hold model copies",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			m, err := clusterMain(cmd)
			if err != nil {
				return err
			}
			locs := m.status.Config.Locations
			if len(locs) == 0 {
				fmt.Println("No locations. Add the fastest first: ollama cluster locations add <dir>")
				return nil
			}
			for i, l := range locs {
				fmt.Printf("%d. %s%s\n", i+1, l.Path, locationNote(l))
			}
			return nil
		},
	}

	var archive, yes bool
	var position int
	add := &cobra.Command{
		Use:   "add <dir>",
		Short: "Add a location (at the end, or at --position; --archive for slow storage)",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			m, err := clusterMain(cmd)
			if err != nil {
				return err
			}
			dir, err := filepath.Abs(args[0])
			if err != nil {
				return err
			}
			if err := os.MkdirAll(filepath.Join(dir, cluster.HFDir), 0o755); err != nil {
				return fmt.Errorf("%s: %w", dir, err)
			}
			cfg := m.config()
			cfg.Locations = slices.DeleteFunc(cfg.Locations, func(l cluster.Location) bool { return l.Path == dir })
			loc := cluster.Location{Path: dir, Archive: archive}
			if position > 0 && position <= len(cfg.Locations) {
				cfg.Locations = slices.Insert(cfg.Locations, position-1, loc)
			} else {
				cfg.Locations = append(cfg.Locations, loc)
			}
			if err := m.api.saveConfig(ctx, cfg); err != nil {
				return err
			}
			for i, l := range cfg.Locations {
				fmt.Printf("%d. %s%s\n", i+1, l.Path, locationNote(l))
			}
			if m.container != nil && !m.container.mounted(dir) {
				m.status.Config = *cfg
				changed, err := updateMainContainer(ctx, m, m.container.Config.Image, addOptions{yes: yes, locationsOnly: true})
				if err != nil {
					return fmt.Errorf("%w; the location is saved, Ollama will see it after 'ollama cluster locations add %s' succeeds", err, dir)
				}
				if changed {
					if err := m.api.waitReady(ctx, 120*time.Second); err != nil {
						return err
					}
					step("Ollama restarted and sees %s", dir)
				}
			}
			return nil
		},
	}
	add.Flags().BoolVar(&archive, "archive", false, "slow storage: keep copies there, never load from it")
	add.Flags().IntVar(&position, "position", 0, "1-based position in the order (default: last)")
	add.Flags().BoolVarP(&yes, "yes", "y", false, "do not ask before restarting Ollama")

	remove := &cobra.Command{
		Use:   "remove <dir>",
		Short: "Forget a location (files stay on disk)",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			m, err := clusterMain(cmd)
			if err != nil {
				return err
			}
			dir, _ := filepath.Abs(args[0])
			cfg := m.config()
			n := len(cfg.Locations)
			cfg.Locations = slices.DeleteFunc(cfg.Locations, func(l cluster.Location) bool { return l.Path == dir })
			if len(cfg.Locations) == n {
				return fmt.Errorf("%s is not a location", dir)
			}
			if err := m.api.saveConfig(ctx, cfg); err != nil {
				return err
			}
			fmt.Printf("%s removed from the locations; the files stay on disk.\n", dir)
			return nil
		},
	}
	cmd.AddCommand(add, remove)
	return cmd
}

// clusterMain detects the main Ollama and requires a cluster build.
func clusterMain(cmd *cobra.Command) (*mainOllama, error) {
	m, err := detectMain(cmd.Context(), cmd)
	if err != nil {
		return nil, err
	}
	if m.status == nil {
		return nil, m.statusErr
	}
	return m, nil
}

// --- pull -------------------------------------------------------------------

type hfFile struct {
	Path string
	Size int64
	SHA  string // sha256 of an LFS file
}

type hfClient struct {
	token string
	http  *http.Client
}

func newHF() *hfClient {
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		home, _ := os.UserHomeDir()
		if data, err := os.ReadFile(filepath.Join(home, ".cache", "huggingface", "token")); err == nil {
			token = strings.TrimSpace(string(data))
		}
	}
	return &hfClient{token: token, http: &http.Client{}}
}

func (h *hfClient) get(ctx context.Context, u string, header http.Header) (*http.Response, error) {
	resp, err := h.do(ctx, u, header, h.token)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode == http.StatusUnauthorized && h.token != "" {
		// a stale token makes even public repositories fail; try without it
		resp.Body.Close()
		if resp, err = h.do(ctx, u, header, ""); err != nil {
			return nil, err
		}
		if resp.StatusCode < 300 {
			fmt.Println("  WARNING: Hugging Face rejected the saved token; continuing without it (public repositories only). Renew it with: hf auth login")
			h.token = ""
		}
	}
	if resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		resp.Body.Close()
		switch resp.StatusCode {
		case http.StatusUnauthorized, http.StatusForbidden:
			return nil, fmt.Errorf("%s: access denied (gated or private repository; set HF_TOKEN)", u)
		case http.StatusNotFound:
			return nil, fmt.Errorf("%s: not found", u)
		}
		return nil, fmt.Errorf("%s: %s %s", u, resp.Status, strings.TrimSpace(string(body)))
	}
	return resp, nil
}

func (h *hfClient) do(ctx context.Context, u string, header http.Header, token string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	for k, v := range header {
		req.Header[k] = v
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	return h.http.Do(req)
}

func (h *hfClient) revision(ctx context.Context, repo string) (string, error) {
	resp, err := h.get(ctx, "https://huggingface.co/api/models/"+repo+"/revision/main", nil)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	var v struct {
		SHA string `json:"sha"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&v); err != nil {
		return "", err
	}
	return v.SHA, nil
}

var linkNext = regexp.MustCompile(`<([^>]+)>;\s*rel="next"`)

func (h *hfClient) tree(ctx context.Context, repo, rev string) ([]hfFile, error) {
	u := "https://huggingface.co/api/models/" + repo + "/tree/" + url.PathEscape(rev) + "?recursive=true"
	var files []hfFile
	for u != "" {
		resp, err := h.get(ctx, u, nil)
		if err != nil {
			return nil, err
		}
		var entries []struct {
			Type string `json:"type"`
			Path string `json:"path"`
			Size int64  `json:"size"`
			LFS  *struct {
				OID string `json:"oid"`
			} `json:"lfs"`
		}
		err = json.NewDecoder(resp.Body).Decode(&entries)
		resp.Body.Close()
		if err != nil {
			return nil, err
		}
		for _, e := range entries {
			// our own manifest and partial downloads are never taken from
			// the repository; paths must stay inside the copy
			if e.Type != "file" || !safeRelPath(e.Path) || path.Base(e.Path) == copyManifestName || strings.HasSuffix(e.Path, ".part") {
				continue
			}
			f := hfFile{Path: e.Path, Size: e.Size}
			if e.LFS != nil {
				f.SHA = e.LFS.OID
			}
			files = append(files, f)
		}
		u = ""
		if m := linkNext.FindStringSubmatch(resp.Header.Get("Link")); m != nil {
			// the token goes only to huggingface.co
			if next, err := url.Parse(m[1]); err == nil && next.Scheme == "https" && next.Host == "huggingface.co" {
				u = next.String()
			}
		}
	}
	return files, nil
}

var shardSuffix = regexp.MustCompile(`(?i)-\d{5}-of-\d{5}$`)

// selectFiles picks the GGUF files of a quantization tag, the best projector
// and the small metadata files.
func selectFiles(files []hfFile, tag string) (model []hfFile, projector *hfFile, extra []hfFile, err error) {
	want := strings.ToUpper(tag)
	groups := map[string][]hfFile{}
	for _, f := range files {
		lower := strings.ToLower(f.Path)
		if !strings.HasSuffix(lower, ".gguf") || strings.Contains(lower, "mmproj") {
			continue
		}
		base := strings.ToUpper(shardSuffix.ReplaceAllString(strings.TrimSuffix(path.Base(f.Path), path.Ext(f.Path)), ""))
		dir := strings.ToUpper(path.Base(path.Dir(f.Path)))
		if strings.HasSuffix(base, "-"+want) || strings.HasSuffix(base, "."+want) || strings.HasSuffix(base, "_"+want) || dir == want {
			key := path.Dir(f.Path) + "/" + base
			groups[key] = append(groups[key], f)
		}
	}
	if len(groups) == 0 {
		return nil, nil, nil, fmt.Errorf("no GGUF file with tag %s", tag)
	}
	keys := make([]string, 0, len(groups))
	for k := range groups {
		keys = append(keys, k)
	}
	// "Q4_K_M" prefers the plain file over "UD-Q4_K_M"; asking for "UD-..." selects it
	sort.Slice(keys, func(i, j int) bool {
		ui, uj := strings.Contains(keys[i], "UD-"+want), strings.Contains(keys[j], "UD-"+want)
		if ui != uj {
			return !ui
		}
		return keys[i] < keys[j]
	})
	model = groups[keys[0]]
	sort.Slice(model, func(i, j int) bool { return model[i].Path < model[j].Path })

	var projs []hfFile
	for _, f := range files {
		lower := strings.ToLower(f.Path)
		if strings.HasSuffix(lower, ".gguf") && strings.Contains(lower, "mmproj") {
			projs = append(projs, f)
		}
		if !strings.Contains(f.Path, "/") && f.Size < 10<<20 &&
			(strings.HasSuffix(lower, ".json") || strings.HasSuffix(lower, ".md") || lower == "license") {
			extra = append(extra, f)
		}
	}
	for _, pref := range []string{"F16", "BF16", "F32", ""} {
		for i := range projs {
			if pref == "" || hasToken(projs[i].Path, pref) {
				projector = &projs[i]
				break
			}
		}
		if projector != nil {
			break
		}
	}
	return model, projector, extra, nil
}

// hasToken reports whether a file name contains tok between separators, so
// "F16" matches mmproj-F16.gguf but not mmproj-BF16.gguf.
func hasToken(name, tok string) bool {
	base := strings.ToUpper(strings.TrimSuffix(path.Base(name), path.Ext(name)))
	return slices.Contains(strings.FieldsFunc(base, func(r rune) bool { return r == '-' || r == '_' || r == '.' }), tok)
}

// parseRepoTag accepts owner/repo:TAG and hf.co/owner/repo:TAG.
func parseRepoTag(s string) (repo, tag string, err error) {
	for _, p := range []string{"https://", "hf.co/", "huggingface.co/"} {
		s = strings.TrimPrefix(s, p)
	}
	repo, tag, _ = strings.Cut(s, ":")
	if strings.Count(repo, "/") != 1 || tag == "" || checkRepo(repo) != nil {
		return "", "", fmt.Errorf("%q: expected owner/repo:TAG, e.g. unsloth/Qwen3-4B-GGUF:Q4_K_M", s)
	}
	return repo, tag, nil
}

func pullCmd() *cobra.Command {
	var name string
	var to string
	var parallel int
	cmd := &cobra.Command{
		Use:   "pull <owner/repo:TAG>",
		Short: "Get a GGUF model from Hugging Face as a repository copy and add it to Ollama",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runPull(cmd, args[0], name, to, parallel)
		},
	}
	cmd.Flags().StringVar(&name, "name", "", "Ollama model name (default: hf.co/<owner>/<repo>:<TAG>)")
	cmd.Flags().StringVar(&to, "to", "", "location to download to (default: the first with enough space)")
	cmd.Flags().IntVar(&parallel, "parallel", 4, "files downloaded at the same time")
	return cmd
}

func runPull(cmd *cobra.Command, ref, name, to string, parallel int) error {
	ctx := cmd.Context()
	start := time.Now()
	repo, tag, err := parseRepoTag(ref)
	if err != nil {
		return err
	}
	m, err := clusterMain(cmd)
	if err != nil {
		return err
	}
	cfg := m.config()
	if len(cfg.LoadLocations()) == 0 {
		return errors.New("no model location configured; add the fastest disk first: ollama cluster locations add <dir>")
	}
	if name == "" {
		name = "hf.co/" + repo + ":" + tag
	}

	hf := newHF()
	rev, err := hf.revision(ctx, repo)
	if err != nil {
		return err
	}
	all, err := hf.tree(ctx, repo, rev)
	if err != nil {
		return err
	}
	modelFiles, projector, extra, err := selectFiles(all, tag)
	if err != nil {
		return fmt.Errorf("%s: %w", repo, err)
	}
	need := append(slices.Clone(modelFiles), extra...)
	if projector != nil {
		need = append(need, *projector)
	}
	files := make([]copyFile, len(need))
	for i, f := range need {
		files[i] = copyFile{Path: f.Path, Size: f.Size, SHA256: f.SHA}
	}
	total := filesSize(files)
	step("%s:%s - %d file(s), %s (revision %s)", repo, tag, len(files), humanBytes(uint64(total)), short(rev))

	// 1. an existing copy of the model files in a load location wins; small
	// files missing there (e.g. after an import) are fetched next to it
	dest := ""
	for _, l := range cfg.LoadLocations() {
		dir := copyDir(l.Path, repo)
		if copyComplete(dir, modelOnly(files)) && !copyVerified(cfg, dir, modelOnly(files)) {
			step("Copy in %s differs from revision %s (another revision or a damaged file); its files are fetched again", l.Path, short(rev))
			continue
		}
		if copyComplete(dir, modelOnly(files)) {
			dest = l.Path
			step("Copy already in %s (checked against the sha256 sums)", l.Path)
			// small files are optional; a copy imported as another user may
			// not be writable
			if err := downloadFiles(ctx, cfg, hf, repo, rev, dir, files, parallel); err != nil {
				fmt.Printf("  WARNING: small repository files not fetched: %v\n", err)
			}
			break
		}
	}

	// 2. otherwise the first load location with space for the rest
	if dest == "" {
		dest, err = pickLocation(cfg, repo, files, to)
		if err != nil {
			return err
		}
		dir := copyDir(dest, repo)
		var archiveSrc string
		for _, l := range cfg.Locations {
			if l.Archive && copyComplete(copyDir(l.Path, repo), modelOnly(files)) {
				archiveSrc = copyDir(l.Path, repo)
			}
		}
		if archiveSrc != "" {
			step("Copying from the archive %s to %s (archive storage is slow)", archiveSrc, dest)
			if err := copyFiles(ctx, cfg, archiveSrc, dir, modelOnly(files)); err != nil {
				return err
			}
		}
		step("Downloading to %s", dir)
		if err := downloadFiles(ctx, cfg, hf, repo, rev, dir, files, parallel); err != nil {
			return err
		}
	}
	dir := copyDir(dest, repo)

	// 3. register in Ollama: link the blobs, then create the model from them
	ollamaFiles := map[string]string{}
	var names []string
	link := slices.Clone(modelFiles)
	if projector != nil {
		link = append(link, *projector)
	}
	for _, f := range link {
		if f.SHA == "" {
			return fmt.Errorf("%s has no sha256 in the repository listing", f.Path)
		}
		digest := "sha256:" + f.SHA
		var lr cluster.LinkResponse
		p := filepath.Join(dir, filepath.FromSlash(f.Path))
		if err := m.api.do(ctx, http.MethodPost, "/api/cluster/link", cluster.LinkRequest{Digest: digest, Path: p}, &lr); err != nil {
			return fmt.Errorf("linking %s: %w", f.Path, err)
		}
		ollamaFiles[path.Base(f.Path)] = digest
		names = append(names, f.Path)
	}
	stream := false
	if err := m.api.do(ctx, http.MethodPost, "/api/create", api.CreateRequest{Model: name, Files: ollamaFiles, Stream: &stream}, nil); err != nil {
		return fmt.Errorf("creating %s in Ollama: %w", name, err)
	}
	if err := mergeCopyManifest(dir, repo, rev, files, name, names); err != nil {
		fmt.Printf("  WARNING: %s not updated: %v\n", copyManifestName, err)
	}
	fmt.Printf("Done in %s: %s uses %s (no second copy in the Ollama store).\n", humanDuration(time.Since(start)), name, dir)
	return nil
}

func short(rev string) string {
	if len(rev) > 10 {
		return rev[:10]
	}
	return rev
}

func modelOnly(files []copyFile) []copyFile {
	return slices.DeleteFunc(slices.Clone(files), func(f copyFile) bool { return f.SHA256 == "" })
}

// pickLocation returns the first load location with room for the missing
// part of a copy plus a reserve.
func pickLocation(cfg *cluster.Config, repo string, files []copyFile, to string) (string, error) {
	var tried []string
	for _, l := range cfg.LoadLocations() {
		if to != "" && l.Path != to {
			continue
		}
		dir := copyDir(l.Path, repo)
		var missing int64
		for _, f := range files {
			fi, err := os.Stat(filepath.Join(dir, filepath.FromSlash(f.Path)))
			if err == nil && fi.Size() == f.Size {
				continue
			}
			missing += f.Size
			if pi, err := os.Stat(filepath.Join(dir, filepath.FromSlash(f.Path)) + ".part"); err == nil {
				missing -= pi.Size()
			}
		}
		free, ok := diskFree(l.Path)
		if !ok {
			tried = append(tried, l.Path+": not available")
			continue
		}
		if uint64(missing)+pullReserve <= free {
			return l.Path, nil
		}
		tried = append(tried, fmt.Sprintf("%s: %s free, needs %s + %s reserve", l.Path, humanBytes(free), humanBytes(uint64(missing)), humanBytes(pullReserve)))
	}
	if to != "" && len(tried) == 0 {
		return "", fmt.Errorf("%s is not a load location (ollama cluster locations)", to)
	}
	return "", fmt.Errorf("no location has enough space:\n  %s\nFree space or add a location: ollama cluster locations add <dir>", strings.Join(tried, "\n  "))
}

type progress struct {
	total int64
	done  atomic.Int64
	start time.Time
}

func (p *progress) run(ctx context.Context, label string) func() {
	ctx, cancel := context.WithCancel(ctx)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		t := time.NewTicker(5 * time.Second)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				done := p.done.Load()
				rate := float64(done) / time.Since(p.start).Seconds()
				eta := ""
				if rate > 0 {
					eta = ", " + humanDuration(time.Duration(float64(p.total-done)/rate)*time.Second) + " left"
				}
				fmt.Printf("  %s %s of %s, %s/s%s\n", label, humanBytes(uint64(done)), humanBytes(uint64(p.total)), humanBytes(uint64(rate)), eta)
			}
		}
	}()
	return func() { cancel(); wg.Wait() }
}

type countingWriter struct {
	w io.Writer
	p *progress
}

func (c countingWriter) Write(b []byte) (int, error) {
	n, err := c.w.Write(b)
	c.p.done.Add(int64(n))
	return n, err
}

// downloadFiles fetches the missing files of a copy, resuming .part files.
// A file already there is kept only when it has the listed size and sha256.
func downloadFiles(ctx context.Context, cfg *cluster.Config, hf *hfClient, repo, rev, dir string, files []copyFile, parallel int) error {
	var todo []copyFile
	for _, f := range files {
		dst := filepath.Join(dir, filepath.FromSlash(f.Path))
		fi, err := os.Stat(dst)
		if err == nil && fi.Size() == f.Size && (f.SHA256 == "" || cluster.HasDigest(cfg, dst, f.SHA256)) {
			continue
		}
		todo = append(todo, f)
	}
	if len(todo) == 0 {
		return nil
	}
	p := &progress{total: filesSize(todo), start: time.Now()}
	stop := p.run(ctx, "downloaded")
	defer stop()

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	sem := make(chan struct{}, max(parallel, 1))
	errs := make(chan error, len(todo))
	var wg sync.WaitGroup
	for _, f := range todo {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			if err := downloadFile(ctx, cfg, hf, repo, rev, dir, f, p); err != nil {
				errs <- fmt.Errorf("%s: %w", f.Path, err)
				cancel()
			}
		}()
	}
	wg.Wait()
	close(errs)
	if err := <-errs; err != nil {
		return fmt.Errorf("%w (run the same pull again to resume)", err)
	}
	return nil
}

func downloadFile(ctx context.Context, cfg *cluster.Config, hf *hfClient, repo, rev, dir string, f copyFile, p *progress) error {
	dst := filepath.Join(dir, filepath.FromSlash(f.Path))
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	part := dst + ".part"
	var offset int64
	if fi, err := os.Stat(part); err == nil && fi.Size() <= f.Size {
		offset = fi.Size()
	}
	p.done.Add(offset)

	// the sum covers the bytes of a resumed .part too, so the finished file
	// is checked as a whole against the listing
	sha := sha256.New()
	if offset > 0 {
		if err := hashInto(sha, part, offset); err != nil {
			return err
		}
	}

	if offset < f.Size || f.Size == 0 {
		h := http.Header{}
		if offset > 0 {
			h.Set("Range", fmt.Sprintf("bytes=%d-", offset))
		}
		u := "https://huggingface.co/" + repo + "/resolve/" + url.PathEscape(rev) + "/" + escapePath(f.Path)
		resp, err := hf.get(ctx, u, h)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		flags := os.O_CREATE | os.O_WRONLY
		if offset > 0 && resp.StatusCode == http.StatusPartialContent {
			flags |= os.O_APPEND
		} else {
			p.done.Add(-offset)
			flags |= os.O_TRUNC
			sha.Reset()
		}
		out, err := os.OpenFile(part, flags, 0o644)
		if err != nil {
			return err
		}
		if _, err := io.Copy(io.MultiWriter(countingWriter{out, p}, sha), resp.Body); err != nil {
			out.Close()
			return err
		}
		if err := out.Close(); err != nil {
			return err
		}
	}
	if fi, err := os.Stat(part); err != nil || (f.Size > 0 && fi.Size() != f.Size) {
		return fmt.Errorf("incomplete download, size %d of %d", sizeOf(fi), f.Size)
	}
	sum := hex.EncodeToString(sha.Sum(nil))
	if f.SHA256 != "" && sum != strings.ToLower(f.SHA256) {
		_ = os.Remove(part)
		return fmt.Errorf("sha256 %s does not match the repository listing %s; the partial file was removed", short(sum), short(f.SHA256))
	}
	if err := os.Rename(part, dst); err != nil {
		return err
	}
	_ = cluster.RecordDigest(cfg, dst, sum)
	return nil
}

// hashInto feeds the first n bytes of a file into h.
func hashInto(h io.Writer, path string, n int64) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.CopyN(h, f, n)
	return err
}

func sizeOf(fi os.FileInfo) int64 {
	if fi == nil {
		return 0
	}
	return fi.Size()
}

func escapePath(p string) string {
	parts := strings.Split(p, "/")
	for i := range parts {
		parts[i] = url.PathEscape(parts[i])
	}
	return strings.Join(parts, "/")
}

// copyFiles copies files between two copies of a repository; a file with a
// listed sha256 is checked while it is copied.
func copyFiles(ctx context.Context, cfg *cluster.Config, srcDir, dstDir string, files []copyFile) error {
	var todo []copyFile
	for _, f := range files {
		if copyVerified(cfg, dstDir, []copyFile{f}) {
			continue
		}
		todo = append(todo, f)
	}
	if len(todo) == 0 {
		return nil
	}
	p := &progress{total: filesSize(todo), start: time.Now()}
	stop := p.run(ctx, "copied")
	defer stop()
	for _, f := range todo {
		if err := ctx.Err(); err != nil {
			return err
		}
		src := filepath.Join(srcDir, filepath.FromSlash(f.Path))
		dst := filepath.Join(dstDir, filepath.FromSlash(f.Path))
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return err
		}
		in, err := os.Open(src)
		if err != nil {
			return err
		}
		out, err := os.Create(dst + ".part")
		if err != nil {
			in.Close()
			return err
		}
		h := sha256.New()
		_, err = io.Copy(io.MultiWriter(countingWriter{out, p}, h), in)
		in.Close()
		if err == nil {
			err = out.Sync()
		}
		if cerr := out.Close(); err == nil {
			err = cerr
		}
		if err != nil {
			return fmt.Errorf("%s: %w", f.Path, err)
		}
		sum := hex.EncodeToString(h.Sum(nil))
		if f.SHA256 != "" && sum != strings.ToLower(f.SHA256) {
			_ = os.Remove(dst + ".part")
			return fmt.Errorf("%s: the source file has sha256 %s, the listing %s; the source copy is damaged or another revision", f.Path, short(sum), short(f.SHA256))
		}
		if err := os.Rename(dst+".part", dst); err != nil {
			return err
		}
		_ = cluster.RecordDigest(cfg, dst, sum)
	}
	return nil
}

// --- models / copy / drop -----------------------------------------------------

type repoCopy struct {
	repo     string
	loc      cluster.Location
	dir      string
	manifest *copyManifest
	size     int64
	complete bool
}

// findCopies lists the repository copies in all locations.
func findCopies(cfg *cluster.Config) []repoCopy {
	var copies []repoCopy
	for _, l := range cfg.Locations {
		owners, _ := os.ReadDir(filepath.Join(l.Path, cluster.HFDir))
		for _, o := range owners {
			if !o.IsDir() {
				continue
			}
			repos, _ := os.ReadDir(filepath.Join(l.Path, cluster.HFDir, o.Name()))
			for _, r := range repos {
				dir := filepath.Join(l.Path, cluster.HFDir, o.Name(), r.Name())
				man, err := readCopyManifest(dir)
				if err != nil {
					continue
				}
				c := repoCopy{repo: o.Name() + "/" + r.Name(), loc: l, dir: dir, manifest: man}
				c.size = filesSize(man.Files)
				c.complete = copyComplete(dir, man.Files)
				copies = append(copies, c)
			}
		}
	}
	return copies
}

func modelsCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "models",
		Short: "List repository copies, their locations and the Ollama models using them",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			m, err := clusterMain(cmd)
			if err != nil {
				return err
			}
			cfg := m.config()
			copies := findCopies(cfg)
			if len(copies) == 0 {
				fmt.Println("No repository copies. Get one with: ollama cluster pull <owner/repo:TAG>")
				return nil
			}
			var links cluster.LinksResponse
			_ = m.api.do(ctx, http.MethodGet, "/api/cluster/links", nil, &links)

			byRepo := map[string][]repoCopy{}
			var repos []string
			for _, c := range copies {
				if _, ok := byRepo[c.repo]; !ok {
					repos = append(repos, c.repo)
				}
				byRepo[c.repo] = append(byRepo[c.repo], c)
			}
			sort.Strings(repos)
			for _, repo := range repos {
				cs := byRepo[repo]
				fmt.Printf("%s  %s\n", repo, humanBytes(uint64(cs[0].size)))
				for _, c := range cs {
					used := 0
					for _, l := range links.Links {
						if strings.HasPrefix(l.Target, c.dir+string(filepath.Separator)) {
							used++
						}
					}
					note := ""
					switch {
					case !c.complete:
						note = "incomplete"
					case c.loc.Archive:
						note = "archive"
					case used > 0:
						note = fmt.Sprintf("Ollama loads from here (%d file(s))", used)
					}
					fmt.Printf("  %-40s %s\n", c.loc.Path, note)
				}
				var models []string
				for name := range cs[0].manifest.Models {
					models = append(models, name)
				}
				sort.Strings(models)
				if len(models) > 0 {
					fmt.Printf("  models: %s\n", strings.Join(models, ", "))
				}
			}
			return nil
		},
	}
}

func copyCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "copy <owner/repo> <location>",
		Short: "Copy a repository to another location (Ollama switches to the best copy)",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			m, err := clusterMain(cmd)
			if err != nil {
				return err
			}
			cfg := m.config()
			repo := strings.TrimPrefix(args[0], "hf.co/")
			dst, _ := filepath.Abs(args[1])
			if !slices.ContainsFunc(cfg.Locations, func(l cluster.Location) bool { return l.Path == dst }) {
				return fmt.Errorf("%s is not a location; add it first: ollama cluster locations add %s", dst, dst)
			}
			src := bestSource(findCopies(cfg), repo, dst)
			if src == nil {
				return fmt.Errorf("no complete copy of %s in any location", repo)
			}
			files := src.manifest.Files
			if free, ok := diskFree(dst); ok && uint64(filesSize(files)) > free {
				return fmt.Errorf("%s has %s free, the copy needs %s", dst, humanBytes(free), humanBytes(uint64(filesSize(files))))
			}
			dstDir := copyDir(dst, repo)
			t := time.Now()
			step("Copying %s (%s) from %s to %s", repo, humanBytes(uint64(filesSize(files))), src.loc.Path, dst)
			if err := copyFiles(ctx, cfg, src.dir, dstDir, files); err != nil {
				return err
			}
			if err := mergeCopyManifest(dstDir, repo, src.manifest.Revision, files, "", nil); err != nil {
				return err
			}
			// carry the model names over
			if man, err := readCopyManifest(dstDir); err == nil {
				for k, v := range src.manifest.Models {
					if man.Models == nil {
						man.Models = map[string][]string{}
					}
					man.Models[k] = v
				}
				_ = man.save(dstDir)
			}
			step("Copied in %s", humanDuration(time.Since(t)))
			if err := relink(ctx, m, ""); err != nil {
				return fmt.Errorf("the copy is complete, but %w", err)
			}
			return nil
		},
	}
}

// bestSource is the fastest complete copy outside dst.
func bestSource(copies []repoCopy, repo, dst string) *repoCopy {
	for i := range copies {
		c := &copies[i]
		if c.repo == repo && c.complete && c.loc.Path != dst {
			return c
		}
	}
	return nil
}

// relink asks Ollama to point every linked blob at the best copy. It returns
// an error when a link pointing into scope (any link when scope is empty)
// could not be changed or has no copy left; problems elsewhere, such as a
// disk of another repository that is not mounted, are only reported.
func relink(ctx context.Context, m *mainOllama, scope string) error {
	var res cluster.RelinkResponse
	if err := m.api.do(ctx, http.MethodPost, "/api/cluster/relink", nil, &res); err != nil {
		return err
	}
	var problems []string
	for _, c := range res.Changes {
		var problem string
		switch {
		case c.Error != "":
			problem = fmt.Sprintf("%s: %s", c.From, c.Error)
		case c.Missing:
			problem = fmt.Sprintf("%s: no copy with this content left", c.From)
		default:
			step("Ollama now loads %s", c.To)
			continue
		}
		if scope == "" || inside(c.From, scope) {
			problems = append(problems, problem)
		} else {
			fmt.Printf("  WARNING: %s\n", problem)
		}
	}
	if len(problems) > 0 {
		return fmt.Errorf("some model files could not be switched:\n  %s", strings.Join(problems, "\n  "))
	}
	return nil
}

// links lists the blobs that Ollama keeps as links into repository copies.
func links(ctx context.Context, m *mainOllama) ([]cluster.Link, error) {
	var res cluster.LinksResponse
	if err := m.api.do(ctx, http.MethodGet, "/api/cluster/links", nil, &res); err != nil {
		return nil, err
	}
	return res.Links, nil
}

// inside reports whether path is dir or below it.
func inside(path, dir string) bool {
	rel, err := filepath.Rel(dir, path)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

// needed is a file of a copy that must exist elsewhere before the copy is
// deleted.
type needed struct {
	rel, digest string
	linked      bool // Ollama loads it, so the other copy must be loadable
}

// uncovered returns the files of target that no other copy has with the same
// content. A file Ollama loads needs a copy in a load location; for the
// others an archive copy is enough.
func uncovered(cfg *cluster.Config, target *repoCopy, others []repoCopy, ls []cluster.Link) []string {
	files := map[string]needed{}
	for _, f := range target.manifest.Files {
		if f.SHA256 != "" {
			files[f.Path] = needed{rel: f.Path, digest: f.SHA256}
		}
	}
	for _, l := range ls {
		if !inside(l.Target, target.dir) {
			continue
		}
		rel, err := filepath.Rel(target.dir, l.Target)
		if err != nil {
			continue
		}
		rel = filepath.ToSlash(rel)
		files[rel] = needed{rel: rel, digest: l.Digest, linked: true}
	}

	var missing []string
	for _, n := range files {
		covered := false
		for _, c := range others {
			if n.linked && c.loc.Archive {
				continue
			}
			if cluster.HasDigest(cfg, filepath.Join(c.dir, filepath.FromSlash(n.rel)), n.digest) {
				covered = true
				break
			}
		}
		if !covered {
			what := n.rel
			if n.linked {
				what += " (Ollama loads it; needs a copy outside archive locations)"
			}
			missing = append(missing, what)
		}
	}
	sort.Strings(missing)
	return missing
}

func dropCmd() *cobra.Command {
	var force, yes bool
	cmd := &cobra.Command{
		Use:   "drop <owner/repo> <location>",
		Short: "Delete a repository copy from one location (never the last one)",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			m, err := clusterMain(cmd)
			if err != nil {
				return err
			}
			cfg := m.config()
			repo := strings.TrimPrefix(args[0], "hf.co/")
			loc, _ := filepath.Abs(args[1])

			var target *repoCopy
			var others []repoCopy
			for _, c := range findCopies(cfg) {
				if c.repo != repo {
					continue
				}
				if c.loc.Path == loc {
					target = &c
				} else {
					// coverage is checked file by file below, so an
					// incomplete copy counts for the files it has
					others = append(others, c)
				}
			}
			if target == nil {
				return fmt.Errorf("no copy of %s in %s", repo, loc)
			}
			if target.manifest == nil {
				target.manifest = &copyManifest{}
			}
			if len(others) == 0 && !force {
				return fmt.Errorf("this is the last copy of %s; models using it would stop working. Use --force to delete it anyway", repo)
			}

			// every file must exist elsewhere with the same content: another
			// copy can hold another quantization or revision of the repository
			ls, err := links(ctx, m)
			if err != nil {
				return err
			}
			step("Checking the other copies against the sha256 of each file (the first check of a copy reads it whole)")
			if missing := uncovered(cfg, target, others, ls); len(missing) > 0 {
				msg := fmt.Sprintf("these files of %s have no other copy with the same content:\n  %s", target.dir, strings.Join(missing, "\n  "))
				if !force {
					return fmt.Errorf("%s\nCopy them first (ollama cluster copy %s <location>) or use --force to delete them anyway", msg, repo)
				}
				fmt.Printf("WARNING: %s\n", msg)
			}
			if !yes && !confirm(fmt.Sprintf("Delete %s (%s)?", target.dir, humanBytes(uint64(target.size))), false) {
				return errors.New("nothing was changed")
			}

			if err := removeCopy(ctx, m, target.dir, force); err != nil {
				return err
			}
			fmt.Printf("Deleted %s, freed %s.\n", target.dir, humanBytes(uint64(target.size)))
			return nil
		},
	}
	cmd.Flags().BoolVar(&force, "force", false, "delete files that have no other copy with the same content")
	cmd.Flags().BoolVarP(&yes, "yes", "y", false, "do not ask for confirmation")
	return cmd
}

// removeCopy moves dir aside, switches Ollama to the other copies and then
// deletes it. When a link cannot be switched, dir is moved back, unless
// force is set.
func removeCopy(ctx context.Context, m *mainOllama, dir string, force bool) error {
	aside := dir + ".dropping"
	if err := os.Rename(dir, aside); err != nil {
		return err
	}
	err := relink(ctx, m, dir)
	if err == nil {
		// no link may still point into the copy being deleted
		if after, lerr := links(ctx, m); lerr != nil {
			err = lerr
		} else {
			for _, l := range after {
				if inside(l.Target, dir) {
					err = fmt.Errorf("Ollama still loads %s from this copy", l.Target)
					break
				}
			}
		}
	}
	if err != nil && !force {
		cause := fmt.Errorf("switching Ollama to another copy failed: %w", err)
		if rerr := os.Rename(aside, dir); rerr != nil {
			return fmt.Errorf("%w; moving the copy back failed too, it is in %s: %v", cause, aside, rerr)
		}
		_ = relink(ctx, m, dir)
		return fmt.Errorf("%w; nothing was deleted", cause)
	}
	if err != nil {
		fmt.Printf("WARNING: %v\n", err)
	}
	return os.RemoveAll(aside)
}

// --- import -----------------------------------------------------------------

var sha256Name = regexp.MustCompile(`^[0-9a-f]{64}$`)

func importCmd() *cobra.Command {
	var to string
	var yes bool
	cmd := &cobra.Command{
		Use:   "import <hf-cache-dir> [owner/repo...]",
		Short: "Turn a Hugging Face cache into repository copies without copying data",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runImport(cmd, args[0], args[1:], to, yes)
		},
	}
	cmd.Flags().StringVar(&to, "to", "", "location on the same disk (default: the configured location on the same disk)")
	cmd.Flags().BoolVarP(&yes, "yes", "y", false, "do not ask for confirmation")
	return cmd
}

type importFile struct {
	path string // repository path
	blob string // cache blob
	dest string
	size int64
	sha  string
	done bool
}

type importRepo struct {
	repo     string
	revision string
	files    []importFile
}

func scanHFCache(cache, loc string, only []string) ([]importRepo, error) {
	entries, err := os.ReadDir(cache)
	if err != nil {
		return nil, err
	}
	realCache, err := filepath.EvalSymlinks(cache)
	if err != nil {
		return nil, err
	}
	var repos []importRepo
	for _, e := range entries {
		if !e.IsDir() || !strings.HasPrefix(e.Name(), "models--") {
			continue
		}
		parts := strings.SplitN(strings.TrimPrefix(e.Name(), "models--"), "--", 2)
		if len(parts) != 2 {
			continue
		}
		repo := parts[0] + "/" + parts[1]
		if len(only) > 0 && !slices.Contains(only, repo) {
			continue
		}
		if checkRepo(repo) != nil {
			continue
		}
		base := filepath.Join(cache, e.Name())
		rev := "main"
		if data, err := os.ReadFile(filepath.Join(base, "refs", "main")); err == nil {
			rev = strings.TrimSpace(string(data))
		}
		snap := filepath.Join(base, "snapshots", rev)
		if _, err := os.Stat(snap); err != nil {
			// no refs/main: take the only snapshot
			snaps, _ := os.ReadDir(filepath.Join(base, "snapshots"))
			if len(snaps) != 1 {
				continue
			}
			rev = snaps[0].Name()
			snap = filepath.Join(base, "snapshots", rev)
		}

		r := importRepo{repo: repo, revision: rev}
		dest := copyDir(loc, repo)
		err := filepath.WalkDir(snap, func(p string, d os.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return err
			}
			rel, _ := filepath.Rel(snap, p)
			target, err := os.Readlink(p)
			if err != nil {
				return nil // not a cache link
			}
			if !filepath.IsAbs(target) {
				target = filepath.Join(filepath.Dir(p), target)
			}
			f := importFile{path: filepath.ToSlash(rel), dest: filepath.Join(dest, rel)}
			// the first link names the file by its sha256; some downloaders
			// add more links, so the data is wherever the chain ends
			if sha256Name.MatchString(filepath.Base(target)) {
				f.sha = filepath.Base(target)
			}
			final, err := filepath.EvalSymlinks(filepath.Clean(target))
			if err != nil {
				return nil // incomplete download
			}
			fi, err := os.Stat(final)
			if err != nil || !fi.Mode().IsRegular() {
				return nil
			}
			f.blob, f.size = final, fi.Size()
			if realDest, err := filepath.EvalSymlinks(f.dest); err == nil && realDest == final {
				f.done = true
			} else if !within(final, realCache) {
				// only data of the cache is moved; a link pointing elsewhere
				// (planted by another user, say) is left alone
				return nil
			}
			r.files = append(r.files, f)
			return nil
		})
		if err != nil {
			return nil, err
		}
		if len(r.files) > 0 {
			repos = append(repos, r)
		}
	}
	return repos, nil
}

func runImport(cmd *cobra.Command, cache string, only []string, to string, yes bool) error {
	cache, err := filepath.Abs(cache)
	if err != nil {
		return err
	}
	if fi, err := os.Stat(filepath.Join(cache, "hub")); err == nil && fi.IsDir() {
		cache = filepath.Join(cache, "hub")
	}

	loc := to
	var m *mainOllama
	if mm, err := detectMain(cmd.Context(), cmd); err == nil && mm.status != nil {
		m = mm
	}
	if loc == "" {
		if m == nil {
			return errors.New("Ollama does not answer; pass the target location with --to")
		}
		for _, l := range m.config().LoadLocations() {
			if sameDisk(l.Path, cache) {
				loc = l.Path
				break
			}
		}
		if loc == "" {
			return fmt.Errorf("no location on the same disk as %s; add one there first (ollama cluster locations add <dir on that disk>) or pass --to", cache)
		}
	}
	loc, _ = filepath.Abs(loc)
	if err := os.MkdirAll(filepath.Join(loc, cluster.HFDir), 0o755); err != nil {
		return err
	}
	if !sameDisk(loc, cache) {
		return fmt.Errorf("%s and %s are on different disks; import moves files and never copies them", loc, cache)
	}

	repos, err := scanHFCache(cache, loc, only)
	if err != nil {
		return err
	}
	if len(repos) == 0 {
		return fmt.Errorf("no complete model repositories in %s", cache)
	}

	var pending int
	for _, r := range repos {
		var size int64
		todo := 0
		for _, f := range r.files {
			size += f.size
			if !f.done {
				todo++
			}
		}
		pending += todo
		state := fmt.Sprintf("%d file(s) to move", todo)
		if todo == 0 {
			state = "already imported"
		}
		fmt.Printf("  %-50s %10s  %s\n", r.repo, humanBytes(uint64(size)), state)
	}
	if pending == 0 {
		fmt.Println("Nothing to do.")
		return registerImported(cmd.Context(), m, loc, repos)
	}
	fmt.Printf("Files move to %s/<owner>/<repo>; %s keeps working through links.\n", filepath.Join(loc, cluster.HFDir), cache)
	if !yes && !confirm("Continue?", false) {
		return errors.New("nothing was changed")
	}

	for _, r := range repos {
		for _, f := range r.files {
			if f.done {
				continue
			}
			if err := os.MkdirAll(filepath.Dir(f.dest), 0o755); err != nil {
				return err
			}
			if _, err := os.Lstat(f.dest); err == nil {
				return fmt.Errorf("%s already exists and is not the cache file; move it away and run import again", f.dest)
			}
			if err := os.Rename(f.blob, f.dest); err != nil {
				return fmt.Errorf("moving %s: %w (run as the owner of the cache, e.g. with sudo)", f.path, err)
			}
			if err := os.Symlink(f.dest, f.blob); err != nil {
				// put the file back so the cache stays intact
				_ = os.Rename(f.dest, f.blob)
				return fmt.Errorf("linking %s back into the cache: %w", f.blob, err)
			}
		}
		step("%s imported", r.repo)
	}
	return registerImported(cmd.Context(), m, loc, repos)
}

// registerImported writes the copy manifests. Models are added to Ollama
// with 'ollama cluster pull', which finds the copy and downloads nothing.
func registerImported(ctx context.Context, m *mainOllama, loc string, repos []importRepo) error {
	for _, r := range repos {
		var files []copyFile
		for _, f := range r.files {
			fi, err := os.Stat(f.dest)
			if err != nil {
				continue
			}
			files = append(files, copyFile{Path: f.path, Size: fi.Size(), SHA256: f.sha})
		}
		dir := copyDir(loc, r.repo)
		if err := mergeCopyManifest(dir, r.repo, r.revision, files, "", nil); err != nil {
			return err
		}
		giveToLocationOwner(loc, dir)
	}
	if m != nil && !slices.ContainsFunc(m.config().Locations, func(l cluster.Location) bool { return l.Path == loc }) {
		fmt.Printf("%s is not a location yet; add it so Ollama can use the copies: ollama cluster locations add %s\n", loc, loc)
	}
	fmt.Println("Add a model from an imported repository with: ollama cluster pull <owner/repo:TAG> (nothing is downloaded)")
	_ = ctx
	return nil
}

// giveToLocationOwner hands the directories and the manifest of a copy made
// by root (an import of a root-owned cache) to the owner of the location, so
// later pulls as that user can add files. The model files stay read-only.
func giveToLocationOwner(loc, dir string) {
	uid, gid, ok := fileOwner(loc)
	if !ok || os.Geteuid() != 0 || uid == 0 {
		return
	}
	_ = filepath.WalkDir(filepath.Join(loc, cluster.HFDir), func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() || d.Name() == copyManifestName {
			_ = os.Lchown(p, uid, gid)
		}
		return nil
	})
}

// within reports whether p is dir or below it.
func within(p, dir string) bool {
	rel, err := filepath.Rel(dir, p)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && !filepath.IsAbs(rel)
}
