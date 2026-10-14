package cluster

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
)

func digestOf(data string) string {
	sum := sha256.Sum256([]byte(data))
	return "sha256:" + hex.EncodeToString(sum[:])
}

func TestLinkAndRelink(t *testing.T) {
	models := t.TempDir()
	t.Setenv("OLLAMA_MODELS", models)
	if err := os.MkdirAll(filepath.Join(models, "blobs"), 0o755); err != nil {
		t.Fatal(err)
	}

	fast, slow, archive := t.TempDir(), t.TempDir(), t.TempDir()
	cfg := &Config{Locations: []Location{{Path: fast}, {Path: slow}, {Path: archive, Archive: true}}}
	rel := filepath.Join(HFDir, "owner", "repo", "model.gguf")
	write := func(root string) string {
		p := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("gguf"), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	slowCopy := write(slow)
	archiveCopy := write(archive)
	digest := digestOf("gguf")

	if _, err := LinkBlob(cfg, digest, archiveCopy); err == nil {
		t.Fatal("linking into an archive location must fail")
	}
	if created, err := LinkBlob(cfg, digest, slowCopy); err != nil || !created {
		t.Fatalf("LinkBlob: created=%v err=%v", created, err)
	}
	if created, err := LinkBlob(cfg, digest, slowCopy); err != nil || created {
		t.Fatalf("second LinkBlob must be a no-op: created=%v err=%v", created, err)
	}

	// a copy in a better location appears: relink prefers it
	fastCopy := write(fast)
	changes, err := Relink(cfg)
	if err != nil || len(changes) != 1 || changes[0].To != fastCopy {
		t.Fatalf("Relink to the faster copy: %+v %v", changes, err)
	}

	// the fast disk disappears: loading repairs the link to the slow copy
	if err := os.RemoveAll(fast); err != nil {
		t.Fatal(err)
	}
	if err := cfg.Save(); err != nil {
		t.Fatal(err)
	}
	blob, _ := blobPath(digest)
	if err := RepairModelLinks(blob); err != nil {
		t.Fatalf("RepairModelLinks: %v", err)
	}
	if target, _ := os.Readlink(blob); target != slowCopy {
		t.Fatalf("blob points at %s, want %s", target, slowCopy)
	}

	// the fast disk comes back: the next load prefers it again
	write(fast)
	if err := RepairModelLinks(blob); err != nil {
		t.Fatalf("RepairModelLinks: %v", err)
	}
	if target, _ := os.Readlink(blob); target != fastCopy {
		t.Fatalf("blob points at %s, want %s", target, fastCopy)
	}

	// no loadable copy left (only the archive): loading fails with a reason
	if err := os.RemoveAll(fast); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(slowCopy); err != nil {
		t.Fatal(err)
	}
	if err := RepairModelLinks(blob); err == nil {
		t.Fatal("RepairModelLinks must fail when only an archive copy is left")
	}
}
