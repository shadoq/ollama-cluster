package clustercmd

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/cluster"
	"github.com/ollama/ollama/types/model"
)

// storeTarget is the name of the Ollama store as the target of mv.
const storeTarget = "store"

func mvCmd() *cobra.Command {
	var yes bool
	cmd := &cobra.Command{
		Use:   "mv <model>... <location|store>",
		Short: "Move the files of models to another location and leave links in Ollama",
		Args:  cobra.MinimumNArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runMove(cmd, args[:len(args)-1], args[len(args)-1], yes)
		},
	}
	cmd.Flags().BoolVarP(&yes, "yes", "y", false, "do not ask for confirmation")
	return cmd
}

// moveBlob is a blob of a model that mv would move.
type moveBlob struct {
	cluster.BlobFile
	models []string // the models using it
}

// movePlan returns the blobs of the named models that are not at dst yet
// (dst empty: the Ollama store) and the reasons the others stay.
func movePlan(cfg *cluster.Config, all []cluster.ModelFiles, names []string, dst string) (map[string][]moveBlob, []string, error) {
	users := map[string][]string{}
	byName := map[string]cluster.ModelFiles{}
	for _, mf := range all {
		byName[mf.Model] = mf
		for _, b := range mf.Blobs {
			users[b.Digest] = append(users[b.Digest], mf.Model)
		}
	}

	plan := map[string][]moveBlob{}
	var notes []string
	for _, name := range names {
		mf, ok := byName[name]
		if !ok {
			return nil, nil, fmt.Errorf("model %s not found (ollama cluster ls)", name)
		}
		for _, b := range mf.Blobs {
			if b.Size < cluster.MinMoveSize && b.OK {
				continue
			}
			place, repo := placeOf(cfg, b.Target)
			switch {
			case b.Target != "" && !b.OK:
				return nil, nil, fmt.Errorf("%s: the file %s is not available; mount its location first (ollama cluster doctor)", name, b.Target)
			case repo != "":
				notes = append(notes, fmt.Sprintf("%s: %s stays in the repository copy hf.co/%s in %s (ollama cluster copy / drop)", name, shortDigest(b.Digest), repo, place))
				continue
			case dst == "" && b.Target == "":
				continue
			case dst != "" && b.Target == cluster.MovedPath(dst, b.Digest):
				continue
			}
			plan[name] = append(plan[name], moveBlob{BlobFile: b, models: users[b.Digest]})
		}
	}
	return plan, notes, nil
}

func runMove(cmd *cobra.Command, args []string, target string, yes bool) error {
	ctx := cmd.Context()
	m, err := clusterMain(cmd)
	if err != nil {
		return err
	}
	cfg := m.config()

	dst, label := "", "the Ollama store"
	if target != storeTarget {
		dst, err = filepath.Abs(target)
		if err != nil {
			return err
		}
		i := slices.IndexFunc(cfg.Locations, func(l cluster.Location) bool { return l.Path == dst })
		switch {
		case i < 0:
			return fmt.Errorf("%s is not a location; add it first: ollama cluster locations add %s", dst, dst)
		case cfg.Locations[i].Archive:
			return fmt.Errorf("%s is an archive location: Ollama never loads from it, so a model moved there could not run", dst)
		case m.container != nil && !m.container.mounted(dst):
			return fmt.Errorf("Ollama in Docker does not see %s; mount it with: ollama cluster locations add %s", dst, dst)
		}
		label = dst
	}

	var names []string
	for _, a := range args {
		n := model.ParseName(a)
		if !n.IsValid() {
			return fmt.Errorf("%q is not a model name", a)
		}
		if name := n.DisplayShortest(); !slices.Contains(names, name) {
			names = append(names, name)
		}
	}

	var files cluster.ModelsResponse
	if err := m.api.do(ctx, http.MethodGet, "/api/cluster/models", nil, &files); err != nil {
		return err
	}
	plan, notes, err := movePlan(cfg, files.Models, names, dst)
	if err != nil {
		return err
	}
	for _, n := range notes {
		fmt.Println("  " + n)
	}

	// a blob shared by several of the named models moves with the first one
	var total int64
	seen := map[string]bool{}
	var shared []string
	for _, name := range names {
		var own []moveBlob
		var size int64
		for _, b := range plan[name] {
			if seen[b.Digest] {
				continue
			}
			seen[b.Digest] = true
			own = append(own, b)
			size += b.Size
			for _, u := range b.models {
				if !slices.Contains(names, u) && !slices.Contains(shared, u) {
					shared = append(shared, u)
				}
			}
		}
		switch {
		case len(own) > 0:
			fmt.Printf("  %-40s %10s  %s -> %s\n", name, humanBytes(uint64(size)), whereNow(cfg, own), label)
		case len(plan[name]) > 0:
			fmt.Printf("  %-40s moves with the models above (same files)\n", name)
		default:
			fmt.Printf("  %-40s already in %s\n", name, label)
		}
		plan[name] = own
		total += size
	}
	if total == 0 {
		fmt.Println("Nothing to move.")
		return nil
	}
	if len(shared) > 0 {
		sort.Strings(shared)
		fmt.Printf("  These models use some of the same files and move with them: %s\n", strings.Join(shared, ", "))
	}
	if dst != "" {
		if free, ok := diskFree(dst); ok && uint64(total) > free {
			return fmt.Errorf("%s has %s free, the move needs %s", dst, humanBytes(free), humanBytes(uint64(total)))
		}
	}
	if !yes && !confirm(fmt.Sprintf("Move %s to %s?", humanBytes(uint64(total)), label), false) {
		return errors.New("nothing was changed")
	}

	t := time.Now()
	for _, name := range names {
		if len(plan[name]) == 0 {
			continue
		}
		if err := moveModel(ctx, m, name, dst, plan[name]); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
	}
	step("Moved in %s", humanDuration(time.Since(t)))
	if dst != "" {
		fmt.Printf("Ollama loads these models from %s; the disk must be mounted for them to run.\n", dst)
	}
	return nil
}

// whereNow names the place the blobs are now: the store, a location or
// several.
func whereNow(cfg *cluster.Config, bs []moveBlob) string {
	var places []string
	for _, b := range bs {
		place, _ := placeOf(cfg, b.Target)
		if place == "store" {
			place = "Ollama store"
		}
		if !slices.Contains(places, place) {
			places = append(places, place)
		}
	}
	return strings.Join(places, ", ")
}

// moveModel asks Ollama to move the blobs of one model and shows the
// progress.
func moveModel(ctx context.Context, m *mainOllama, name, dst string, bs []moveBlob) error {
	var total int64
	for _, b := range bs {
		total += b.Size
	}
	p := &progress{total: total, start: time.Now()}
	stop := p.run(ctx, "moved")
	defer stop()

	var finished int64
	step("Moving %s (%s)", name, humanBytes(uint64(total)))
	return m.api.stream(ctx, http.MethodPost, "/api/cluster/move", cluster.MoveRequest{Model: name, To: dst}, func(line json.RawMessage) error {
		var ev cluster.MoveProgress
		if err := json.Unmarshal(line, &ev); err != nil {
			return err
		}
		switch ev.Status {
		case "moving":
			p.done.Store(finished + ev.Completed)
		case "moved":
			finished += ev.Total
			p.done.Store(finished)
		}
		return nil
	})
}

// unusedMoved returns the moved blobs in the locations that no link points
// at anymore, and their size.
func unusedMoved(cfg *cluster.Config, ls []cluster.Link) ([]string, int64) {
	used := map[string]bool{}
	for _, l := range ls {
		used[filepath.Clean(l.Target)] = true
	}
	var unused []string
	var size int64
	for _, loc := range cfg.Locations {
		entries, _ := os.ReadDir(filepath.Join(loc.Path, cluster.MovedDir))
		for _, e := range entries {
			if !e.Type().IsRegular() || !sha256Blob.MatchString(e.Name()) {
				continue
			}
			p := filepath.Join(loc.Path, cluster.MovedDir, e.Name())
			if used[p] {
				continue
			}
			if fi, err := e.Info(); err == nil {
				size += fi.Size()
			}
			unused = append(unused, p)
		}
	}
	return unused, size
}

var sha256Blob = regexp.MustCompile(`^sha256-[0-9a-f]{64}$`)
