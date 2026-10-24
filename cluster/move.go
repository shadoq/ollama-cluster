package cluster

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// Blobs of models from the Ollama library can be moved out of the Ollama
// store to a location (<location>/ollama/blobs/sha256-<hex>) and left behind
// as links, the same way blobs of repository copies are linked. A model that
// is rarely used then takes no space on the fast disk, and the links follow
// the usual rules: a broken one switches to another location that has the
// file, and a load fails with a clear error when none has it.

// MovedDir is the directory inside a location that holds moved blobs.
var MovedDir = filepath.Join("ollama", "blobs")

// MinMoveSize is the smallest blob that is moved; smaller ones (template,
// parameters, license, config) stay in the store.
const MinMoveSize = 16 << 20

// MoveRequest is the body of POST /api/cluster/move. To is a load location,
// or empty for the Ollama store.
type MoveRequest struct {
	Model string `json:"model"`
	To    string `json:"to"`
}

// MoveProgress is streamed by POST /api/cluster/move: one line per blob
// started or finished and progress while one is copied. The last line has
// Status "success" or an error.
type MoveProgress struct {
	Status    string      `json:"status"`
	Digest    string      `json:"digest,omitempty"`
	Total     int64       `json:"total,omitempty"`
	Completed int64       `json:"completed,omitempty"`
	Result    *MoveResult `json:"result,omitempty"`
}

// MoveResult reports what happened to one blob. From and To are paths, or
// empty for the store. Skipped explains why the blob stayed where it was.
type MoveResult struct {
	Digest  string `json:"digest"`
	Size    int64  `json:"size"`
	From    string `json:"from,omitempty"`
	To      string `json:"to,omitempty"`
	Copied  bool   `json:"copied,omitempty"` // false: moved within the disk
	Skipped string `json:"skipped,omitempty"`
}

// MovedPath is where a moved blob lives in a location.
func MovedPath(loc, digest string) string {
	return filepath.Join(loc, MovedDir, strings.Replace(digest, ":", "-", 1))
}

// isMoved reports whether path is a moved blob in one of the locations.
func isMoved(cfg *Config, path string) bool {
	loc, rel := locationOf(cfg.Locations, path)
	return loc != nil && filepath.Dir(rel) == MovedDir
}

// MoveBlob moves the data of the blob digest to the location dst, or back to
// the Ollama store when dst is empty, and leaves the blob as a link to it.
// The new file is checked against the digest while it is copied; within one
// disk it is hard linked instead. The old file is removed only after the
// blob points at the new one, and only when it is a moved blob: files of
// repository copies stay (ollama cluster copy and drop manage them).
func MoveBlob(ctx context.Context, cfg *Config, digest, dst string, progress func(done int64)) (MoveResult, error) {
	res := MoveResult{Digest: digest}
	blob, err := blobPath(digest)
	if err != nil {
		return res, err
	}
	li, err := os.Lstat(blob)
	if err != nil {
		return res, err
	}
	if li.Mode()&os.ModeSymlink != 0 {
		target, err := os.Readlink(blob)
		if err != nil {
			return res, err
		}
		if !filepath.IsAbs(target) {
			target = filepath.Join(filepath.Dir(blob), target)
		}
		res.From = filepath.Clean(target)
		if !isMoved(cfg, res.From) {
			res.Skipped = "a file of a repository copy (ollama cluster copy / drop)"
			return res, nil
		}
	}
	fi, err := os.Stat(blob)
	if err != nil {
		return res, fmt.Errorf("%s: %w (the location holding it is not available)", res.From, err)
	}
	res.Size = fi.Size()

	if dst == "" {
		if res.From == "" {
			res.Skipped = "already in the Ollama store"
			return res, nil
		}
		// the new file is renamed over the link
		tmp := blob + ".move-part"
		copied, err := placeFile(ctx, res.From, tmp, digest, progress)
		if err != nil {
			return res, err
		}
		if err := os.Rename(tmp, blob); err != nil {
			_ = os.Remove(tmp)
			return res, err
		}
		res.Copied = copied
		return res, os.Remove(res.From)
	}

	dst = filepath.Clean(dst)
	var loc *Location
	for i := range cfg.Locations {
		if filepath.Clean(cfg.Locations[i].Path) == dst {
			loc = &cfg.Locations[i]
		}
	}
	if loc == nil {
		return res, fmt.Errorf("%s is not a location", dst)
	}
	if loc.Archive {
		return res, fmt.Errorf("%s is an archive location: Ollama never loads from it", dst)
	}
	to := MovedPath(loc.Path, digest)
	if res.From == to {
		res.Skipped = "already there"
		return res, nil
	}
	res.To = to
	if err := os.MkdirAll(filepath.Dir(to), 0o755); err != nil {
		return res, err
	}

	// a file left by an earlier move is reused when it has the content
	if !HasDigest(cfg, to, digest) {
		_ = os.Remove(to)
		tmp := to + ".part"
		copied, err := placeFile(ctx, blob, tmp, digest, progress)
		if err != nil {
			return res, err
		}
		if err := os.Rename(tmp, to); err != nil {
			_ = os.Remove(tmp)
			return res, err
		}
		res.Copied = copied
		_ = RecordDigest(cfg, to, digest)
	}
	if err := replaceSymlink(to, blob); err != nil {
		return res, err
	}
	if res.From != "" {
		return res, os.Remove(res.From)
	}
	return res, nil
}

// placeFile makes dst a file with the content of src: a hard link when both
// are on one disk (the store verified the content when it wrote the blob),
// otherwise a copy checked against digest while it is written. It reports
// whether the data was copied.
func placeFile(ctx context.Context, src, dst, digest string, progress func(done int64)) (bool, error) {
	_ = os.Remove(dst)
	real, err := filepath.EvalSymlinks(src)
	if err != nil {
		return false, err
	}
	if linkFile(real, dst) == nil {
		return false, nil
	}

	in, err := os.Open(real)
	if err != nil {
		return false, err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return false, err
	}
	h := sha256.New()
	w := &progressWriter{ctx: ctx, w: io.MultiWriter(out, h), progress: progress}
	_, err = io.Copy(w, in)
	if err == nil {
		err = out.Sync()
	}
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	if err == nil && "sha256:"+hex.EncodeToString(h.Sum(nil)) != NormalizeDigest(digest) {
		err = fmt.Errorf("%s does not have the content of %s; nothing was moved", src, digest)
	}
	if err != nil {
		_ = os.Remove(dst)
		return true, err
	}
	return true, nil
}

// linkFile is os.Link; tests replace it to take the copy path.
var linkFile = os.Link

type progressWriter struct {
	ctx      context.Context
	w        io.Writer
	done     int64
	progress func(int64)
}

func (p *progressWriter) Write(b []byte) (int, error) {
	if err := p.ctx.Err(); err != nil {
		return 0, errors.Join(errors.New("the move was interrupted; nothing was changed"), err)
	}
	n, err := p.w.Write(b)
	p.done += int64(n)
	if p.progress != nil {
		p.progress(p.done)
	}
	return n, err
}
