package cluster

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestFileDigestCache(t *testing.T) {
	loc := t.TempDir()
	cfg := &Config{Locations: []Location{{Path: loc}}}
	p := filepath.Join(loc, HFDir, "owner", "repo", "model.gguf")
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte("first"), 0o644); err != nil {
		t.Fatal(err)
	}

	d, err := FileDigest(cfg, p)
	if err != nil || d != digestOf("first") {
		t.Fatalf("FileDigest = %s, %v", d, err)
	}
	if !HasDigest(cfg, p, d[len("sha256:"):]) {
		t.Fatal("a bare hex sum (as Hugging Face lists it) must match")
	}

	// the recorded digest is used while size and modification time match
	cache := filepath.Join(loc, digestCacheName)
	m := readDigestCache(cache)
	e := m["hf/owner/repo/model.gguf"]
	if e.Digest != d {
		t.Fatalf("cache entry %+v", e)
	}
	e.Digest = digestOf("recorded")
	m["hf/owner/repo/model.gguf"] = e
	data, _ := json.Marshal(m)
	if err := os.WriteFile(cache, data, 0o644); err != nil {
		t.Fatal(err)
	}
	if d, _ := FileDigest(cfg, p); d != digestOf("recorded") {
		t.Fatalf("the recorded digest must be used for an unchanged file, got %s", d)
	}

	// same size, other content and modification time: hashed again
	if err := os.WriteFile(p, []byte("other"), 0o644); err != nil {
		t.Fatal(err)
	}
	later := time.Now().Add(time.Minute)
	if err := os.Chtimes(p, later, later); err != nil {
		t.Fatal(err)
	}
	if d, _ := FileDigest(cfg, p); d != digestOf("other") {
		t.Fatalf("a changed file must be hashed again, got %s", d)
	}
}

func TestLinkRequiresSameContent(t *testing.T) {
	models := t.TempDir()
	t.Setenv("OLLAMA_MODELS", models)
	if err := os.MkdirAll(filepath.Join(models, "blobs"), 0o755); err != nil {
		t.Fatal(err)
	}

	fast, slow := t.TempDir(), t.TempDir()
	cfg := &Config{Locations: []Location{{Path: fast}, {Path: slow}}}
	rel := filepath.Join(HFDir, "owner", "repo", "model-Q4.gguf")
	write := func(root, content string) string {
		p := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	digest := digestOf("revision-1")
	slowCopy := write(slow, "revision-1")

	// a file with the same name and size but another revision is not linked
	if _, err := LinkBlob(cfg, digest, write(fast, "revision-2")); err == nil {
		t.Fatal("LinkBlob must refuse a file with other content")
	}
	if _, err := LinkBlob(cfg, digest, slowCopy); err != nil {
		t.Fatal(err)
	}

	// the faster location has another revision of the same size: no switch
	changes, err := Relink(cfg)
	if err != nil || len(changes) != 0 {
		t.Fatalf("Relink must keep the link: %+v %v", changes, err)
	}

	// the original disappears and the only other file is truncated: the
	// link is reported missing instead of pointing at the broken file
	write(fast, "revision")
	if err := os.Remove(slowCopy); err != nil {
		t.Fatal(err)
	}
	changes, err = Relink(cfg)
	if err != nil || len(changes) != 1 || !changes[0].Missing || changes[0].To != "" {
		t.Fatalf("Relink must report the file missing: %+v %v", changes, err)
	}

	// a complete copy with the right content is accepted again
	fastCopy := write(fast, "revision-1")
	changes, err = Relink(cfg)
	if err != nil || len(changes) != 1 || changes[0].To != fastCopy {
		t.Fatalf("Relink to the verified copy: %+v %v", changes, err)
	}
}
