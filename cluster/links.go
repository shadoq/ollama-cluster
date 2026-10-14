package cluster

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/manifest"
)

// Models from Hugging Face are kept as plain repository copies
// (<location>/hf/<owner>/<repo>/<files>) and registered in Ollama through
// symlinks: Hugging Face names every large file by its sha256 and Ollama names
// blobs the same way, so blobs/sha256-<sum> can point at the file in the copy.
// Ollama then stores only its small manifest and config, and llama.cpp, vLLM
// or llama-swap keep using the same files.

// HFDir is the directory inside a location that holds repository copies.
const HFDir = "hf"

// Link is a blob that points at a file outside the Ollama blob store.
type Link struct {
	Digest string `json:"digest"`
	Target string `json:"target"`
	OK     bool   `json:"ok"`
}

// RelinkResult reports what Relink changed.
type RelinkResult struct {
	Digest string `json:"digest"`
	From   string `json:"from"`
	To     string `json:"to,omitempty"`

	// Missing is set when no location has a file with this content anymore.
	Missing bool `json:"missing,omitempty"`

	// Error is set when a copy was found but the link could not be changed.
	Error string `json:"error,omitempty"`
}

// blobPath returns the blob path for a digest of the form sha256:<hex>.
func blobPath(digest string) (string, error) {
	return manifest.BlobsPath(digest)
}

// LinkBlob makes the blob for digest a symlink to target. target must be a
// regular file inside one of the configured load locations and must have the
// content of digest. An existing regular blob is kept: it already holds the
// data.
func LinkBlob(cfg *Config, digest, target string) (created bool, err error) {
	target = filepath.Clean(target)
	if !filepath.IsAbs(target) {
		return false, fmt.Errorf("%s: path must be absolute", target)
	}
	loc, _ := locationOf(cfg.Locations, target)
	if loc == nil {
		return false, fmt.Errorf("%s is not inside a configured location", target)
	}
	if loc.Archive {
		return false, fmt.Errorf("%s is in an archive location; copy it to a faster location first", target)
	}

	fi, err := os.Stat(target)
	if err != nil {
		return false, err
	}
	if !fi.Mode().IsRegular() {
		return false, fmt.Errorf("%s is not a regular file", target)
	}
	// a link inside the location must not lead out of it
	real, err := filepath.EvalSymlinks(target)
	if err != nil {
		return false, err
	}
	realLoc, err := filepath.EvalSymlinks(loc.Path)
	if err != nil {
		return false, err
	}
	if rel, err := filepath.Rel(realLoc, real); err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return false, fmt.Errorf("%s leads out of the location %s", target, loc.Path)
	}

	blob, err := blobPath(digest)
	if err != nil {
		return false, err
	}

	if li, err := os.Lstat(blob); err == nil {
		if li.Mode()&os.ModeSymlink == 0 {
			return false, nil
		}
		if current, err := os.Readlink(blob); err == nil && current == target {
			return false, nil
		}
	}

	if !HasDigest(cfg, target, digest) {
		return false, fmt.Errorf("%s does not have the content of %s (another revision or an incomplete file)", target, digest)
	}
	return true, replaceSymlink(target, blob)
}

// replaceSymlink atomically points link at target.
func replaceSymlink(target, link string) error {
	tmp := link + ".link-tmp"
	_ = os.Remove(tmp)
	if err := os.Symlink(target, tmp); err != nil {
		return err
	}
	return os.Rename(tmp, link)
}

// Links lists the blobs that are symlinks.
func Links() ([]Link, error) {
	dir, err := blobPath("")
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	var links []Link
	for _, e := range entries {
		if e.Type()&os.ModeSymlink == 0 || !strings.HasPrefix(e.Name(), "sha256-") {
			continue
		}
		target, err := os.Readlink(filepath.Join(dir, e.Name()))
		if err != nil {
			continue
		}
		_, statErr := os.Stat(target)
		links = append(links, Link{
			Digest: strings.Replace(e.Name(), "-", ":", 1),
			Target: target,
			OK:     statErr == nil,
		})
	}
	return links, nil
}

// locationOf returns the location containing path and the path relative to it.
func locationOf(locs []Location, path string) (*Location, string) {
	path = filepath.Clean(path)
	var best *Location
	var bestRel string
	for i := range locs {
		root := filepath.Clean(locs[i].Path)
		rel, err := filepath.Rel(root, path)
		if err != nil || rel == "." || strings.HasPrefix(rel, "..") {
			continue
		}
		// prefer the most specific location when locations nest
		if best == nil || len(root) > len(filepath.Clean(best.Path)) {
			best, bestRel = &locs[i], rel
		}
	}
	return best, bestRel
}

// BestCopy returns the file rel in the first load location (in configured
// order) whose content has digest. current, a copy already in use, is
// trusted without hashing it again. size 0 skips the quick size check.
func BestCopy(cfg *Config, rel, digest, current string, size int64) (string, bool) {
	for _, loc := range cfg.LoadLocations() {
		p := filepath.Join(loc.Path, rel)
		fi, err := os.Stat(p)
		if err != nil || !fi.Mode().IsRegular() {
			continue
		}
		if size > 0 && fi.Size() != size {
			continue
		}
		if p != current && !HasDigest(cfg, p, digest) {
			slog.Warn("cluster: copy skipped, its content differs from the model file", "path", p, "digest", digest)
			continue
		}
		return p, true
	}
	return "", false
}

// relinkOne points one symlink blob at the best available copy.
func relinkOne(cfg *Config, l Link) (RelinkResult, bool) {
	res := RelinkResult{Digest: l.Digest, From: l.Target}

	loc, rel := locationOf(cfg.Locations, l.Target)
	if loc == nil {
		// not one of ours (or the location was removed from the config)
		return res, false
	}

	var size int64
	current := ""
	if fi, err := os.Stat(l.Target); err == nil {
		size = fi.Size()
		current = filepath.Clean(l.Target)
	}
	best, ok := BestCopy(cfg, rel, l.Digest, current, size)
	if !ok {
		if !l.OK {
			res.Missing = true
			return res, true
		}
		return res, false
	}
	if best == l.Target {
		return res, false
	}

	blob, err := blobPath(l.Digest)
	if err != nil {
		res.Error = err.Error()
		return res, true
	}
	if err := replaceSymlink(best, blob); err != nil {
		slog.Warn("cluster: relink failed", "digest", l.Digest, "to", best, "error", err)
		res.Error = err.Error()
		return res, true
	}
	res.To = best
	return res, true
}

// Relink points every symlinked blob at the copy in the best available
// location: a copy that disappeared (unmounted disk, powered-off node) is
// replaced by the next one in order, and a copy in a better location is
// preferred once it appears.
func Relink(cfg *Config) ([]RelinkResult, error) {
	links, err := Links()
	if err != nil {
		return nil, err
	}
	var results []RelinkResult
	for _, l := range links {
		if res, changed := relinkOne(cfg, l); changed {
			results = append(results, res)
		}
	}
	return results, nil
}

// RepairModelLinks checks the blobs of a model right before it loads: a
// broken symlink switches to another copy, and a copy in a better location
// (a disk that came back) is preferred. It returns an error naming the
// missing file when no location has it.
func RepairModelLinks(paths ...string) error {
	cfg, err := LoadConfig()
	if err != nil || len(cfg.Locations) == 0 {
		return nil
	}

	for _, p := range paths {
		if p == "" {
			continue
		}
		li, err := os.Lstat(p)
		if err != nil || li.Mode()&os.ModeSymlink == 0 {
			continue
		}
		target, err := os.Readlink(p)
		if err != nil {
			continue
		}
		_, statErr := os.Stat(target)

		digest := strings.Replace(filepath.Base(p), "-", ":", 1)
		res, _ := relinkOne(cfg, Link{Digest: digest, Target: target, OK: statErr == nil})
		switch {
		case res.To != "":
			slog.Info("cluster: model file now loads from another location", "from", res.From, "to", res.To)
		case statErr != nil:
			return fmt.Errorf("model file %s is not available: the location holding it is missing and no other location has a copy", target)
		}
	}
	return nil
}
