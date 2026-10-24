package clustercmd

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/cluster"
)

func TestMovePlan(t *testing.T) {
	loc, other := t.TempDir(), t.TempDir()
	cfg := &cluster.Config{Locations: []cluster.Location{{Path: loc}, {Path: other}}}
	const big = cluster.MinMoveSize
	all := []cluster.ModelFiles{
		{Model: "qwen3.5:9b", Blobs: []cluster.BlobFile{
			{Digest: "sha256:aa", Size: big, OK: true},
			{Digest: "sha256:tt", Size: 100, OK: true}, // template: stays
			{Digest: "sha256:bb", Size: big, OK: true, Target: cluster.MovedPath(loc, "sha256:bb")},
		}},
		{Model: "qwen3.5:9b-q8", Blobs: []cluster.BlobFile{
			{Digest: "sha256:aa", Size: big, OK: true},
		}},
		{Model: "hf.co/owner/repo:Q4_K_M", Blobs: []cluster.BlobFile{
			{Digest: "sha256:cc", Size: big, OK: true, Target: filepath.Join(other, cluster.HFDir, "owner", "repo", "m.gguf")},
		}},
	}

	plan, notes, err := movePlan(cfg, all, []string{"qwen3.5:9b"}, loc)
	if err != nil {
		t.Fatal(err)
	}
	if got := plan["qwen3.5:9b"]; len(got) != 1 || got[0].Digest != "sha256:aa" || len(got[0].models) != 2 {
		t.Fatalf("to a location: only the large blob not there yet, shared by both models: %+v", got)
	}
	if len(notes) != 0 {
		t.Fatalf("notes: %v", notes)
	}

	plan, _, err = movePlan(cfg, all, []string{"qwen3.5:9b"}, "")
	if err != nil {
		t.Fatal(err)
	}
	if got := plan["qwen3.5:9b"]; len(got) != 1 || got[0].Digest != "sha256:bb" {
		t.Fatalf("to the store: only the moved blob: %+v", got)
	}

	plan, notes, err = movePlan(cfg, all, []string{"hf.co/owner/repo:Q4_K_M"}, loc)
	if err != nil || len(plan["hf.co/owner/repo:Q4_K_M"]) != 0 || len(notes) != 1 {
		t.Fatalf("files of a repository copy stay: %+v %v %v", plan, notes, err)
	}

	if _, _, err := movePlan(cfg, all, []string{"missing:1b"}, loc); err == nil {
		t.Fatal("an unknown model must fail")
	}

	all[0].Blobs[2].OK = false
	if _, _, err := movePlan(cfg, all, []string{"qwen3.5:9b"}, ""); err == nil {
		t.Fatal("a model with an unreachable file must fail")
	}
}

func TestUnusedMoved(t *testing.T) {
	loc := t.TempDir()
	cfg := &cluster.Config{Locations: []cluster.Location{{Path: loc}}}
	used := cluster.MovedPath(loc, "sha256:"+sha64('a'))
	unused := cluster.MovedPath(loc, "sha256:"+sha64('b'))
	if err := os.MkdirAll(filepath.Dir(used), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{used, unused, unused + ".part"} {
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	got, size := unusedMoved(cfg, []cluster.Link{{Target: used}})
	if len(got) != 1 || got[0] != unused || size != 1 {
		t.Fatalf("unused: %v %d", got, size)
	}
}

func sha64(c byte) string {
	b := make([]byte, 64)
	for i := range b {
		b[i] = c
	}
	return string(b)
}
