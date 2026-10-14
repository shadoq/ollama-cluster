package cluster

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// A file in a repository copy is used for a blob only when its content has
// the blob's sha256: a copy can hold another revision of a file under the
// same name and size, or a truncated file. Hashing a model file takes a
// minute or more, so a location keeps the digests it has computed, keyed by
// path, size and modification time, and an unchanged file is read once.

// digestCacheName is the file in a location root that keeps computed digests.
const digestCacheName = ".ollama-cluster-digests.json"

type digestEntry struct {
	Size    int64  `json:"size"`
	ModTime int64  `json:"mtime_ns"`
	Digest  string `json:"digest"`
}

var digestMu sync.Mutex

// NormalizeDigest returns a digest as sha256:<hex>; a bare hex sum (as Hugging
// Face lists it) gets the prefix.
func NormalizeDigest(d string) string {
	d = strings.ToLower(strings.TrimSpace(d))
	if d != "" && !strings.HasPrefix(d, "sha256:") {
		d = "sha256:" + d
	}
	return d
}

func digestCacheFor(cfg *Config, path string) (file, rel string, ok bool) {
	if cfg == nil {
		return "", "", false
	}
	loc, rel := locationOf(cfg.Locations, path)
	if loc == nil {
		return "", "", false
	}
	return filepath.Join(loc.Path, digestCacheName), filepath.ToSlash(rel), true
}

func readDigestCache(file string) map[string]digestEntry {
	m := map[string]digestEntry{}
	if data, err := os.ReadFile(file); err == nil {
		_ = json.Unmarshal(data, &m)
	}
	return m
}

// cachedDigest returns the recorded digest of path when its size and
// modification time still match.
func cachedDigest(cfg *Config, path string, fi os.FileInfo) (string, bool) {
	file, rel, ok := digestCacheFor(cfg, path)
	if !ok {
		return "", false
	}
	digestMu.Lock()
	defer digestMu.Unlock()
	e, ok := readDigestCache(file)[rel]
	if !ok || e.Size != fi.Size() || e.ModTime != fi.ModTime().UnixNano() {
		return "", false
	}
	return e.Digest, true
}

// RecordDigest stores the digest of path, computed by the caller (for
// example while downloading). Paths outside the locations are not recorded;
// a location that cannot be written (an archive mounted read-only) only
// loses the shortcut.
func RecordDigest(cfg *Config, path, digest string) error {
	file, rel, ok := digestCacheFor(cfg, path)
	if !ok {
		return nil
	}
	fi, err := os.Stat(path)
	if err != nil {
		return err
	}
	digestMu.Lock()
	defer digestMu.Unlock()
	m := readDigestCache(file)
	m[rel] = digestEntry{Size: fi.Size(), ModTime: fi.ModTime().UnixNano(), Digest: NormalizeDigest(digest)}
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	tmp := fmt.Sprintf("%s.%d.tmp", file, os.Getpid())
	if err := os.WriteFile(tmp, append(data, '\n'), 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, file); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

// FileDigest returns the sha256 of a regular file as sha256:<hex>, reading
// the file only when no digest is recorded for its current size and
// modification time.
func FileDigest(cfg *Config, path string) (string, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	if !fi.Mode().IsRegular() {
		return "", fmt.Errorf("%s is not a regular file", path)
	}
	if d, ok := cachedDigest(cfg, path, fi); ok {
		return d, nil
	}

	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	d := "sha256:" + hex.EncodeToString(h.Sum(nil))

	// record it only when the file did not change while it was read
	if after, err := os.Stat(path); err == nil && after.Size() == fi.Size() && after.ModTime().Equal(fi.ModTime()) {
		_ = RecordDigest(cfg, path, d)
	}
	return d, nil
}

// HasDigest reports whether the file at path has the content of digest.
func HasDigest(cfg *Config, path, digest string) bool {
	d, err := FileDigest(cfg, path)
	return err == nil && d == NormalizeDigest(digest)
}
