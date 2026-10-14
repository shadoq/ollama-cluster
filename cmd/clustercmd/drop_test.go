package clustercmd

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/cluster"
)

func sum(data string) string {
	s := sha256.Sum256([]byte(data))
	return hex.EncodeToString(s[:])
}

// writeCopy creates a repository copy with the given files (name -> content).
func writeCopy(t *testing.T, loc cluster.Location, files map[string]string) repoCopy {
	t.Helper()
	dir := copyDir(loc.Path, "owner/repo")
	man := &copyManifest{Repo: "owner/repo"}
	for name, content := range files {
		p := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		man.Files = append(man.Files, copyFile{Path: name, Size: int64(len(content)), SHA256: sum(content)})
	}
	return repoCopy{repo: "owner/repo", loc: loc, dir: dir, manifest: man}
}

func TestDropNeedsEveryFileElsewhere(t *testing.T) {
	fast := cluster.Location{Path: t.TempDir()}
	slow := cluster.Location{Path: t.TempDir()}
	other2 := cluster.Location{Path: t.TempDir()}
	archive := cluster.Location{Path: t.TempDir(), Archive: true}
	cfg := &cluster.Config{Locations: []cluster.Location{fast, slow, other2, archive}}

	// the copy to delete has two quantizations, the other copy only one
	target := writeCopy(t, fast, map[string]string{"model-Q4.gguf": "q4", "model-Q8.gguf": "q8"})
	other := writeCopy(t, slow, map[string]string{"model-Q4.gguf": "q4"})
	missing := uncovered(cfg, &target, []repoCopy{other}, nil)
	if len(missing) != 1 || missing[0] != "model-Q8.gguf" {
		t.Fatalf("uncovered = %v, want model-Q8.gguf", missing)
	}

	// a file of the same name and size with other content does not count
	other = writeCopy(t, other2, map[string]string{"model-Q4.gguf": "q4", "model-Q8.gguf": "x8"})
	if missing := uncovered(cfg, &target, []repoCopy{other}, nil); len(missing) != 1 {
		t.Fatalf("uncovered = %v, want model-Q8.gguf", missing)
	}

	// an archive copy keeps a file, but a file Ollama loads needs a loadable copy
	arch := writeCopy(t, archive, map[string]string{"model-Q4.gguf": "q4", "model-Q8.gguf": "q8"})
	if missing := uncovered(cfg, &target, []repoCopy{arch}, nil); len(missing) != 0 {
		t.Fatalf("archive copy must cover unlinked files: %v", missing)
	}
	linked := []cluster.Link{{Digest: "sha256:" + sum("q8"), Target: filepath.Join(target.dir, "model-Q8.gguf"), OK: true}}
	missing = uncovered(cfg, &target, []repoCopy{arch}, linked)
	if len(missing) != 1 || !strings.HasPrefix(missing[0], "model-Q8.gguf") {
		t.Fatalf("uncovered = %v, want the linked model-Q8.gguf", missing)
	}

	full := writeCopy(t, slow, map[string]string{"model-Q4.gguf": "q4", "model-Q8.gguf": "q8"})
	if missing := uncovered(cfg, &target, []repoCopy{full}, linked); len(missing) != 0 {
		t.Fatalf("uncovered = %v, want none", missing)
	}
}

// fakeOllama answers the link endpoints of the main Ollama.
func fakeOllama(t *testing.T, relink cluster.RelinkResponse, links func() []cluster.Link) *mainOllama {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/cluster/relink":
			_ = json.NewEncoder(w).Encode(relink)
		case "/api/cluster/links":
			_ = json.NewEncoder(w).Encode(cluster.LinksResponse{Links: links()})
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	return &mainOllama{api: &apiClient{base: srv.URL, http: srv.Client()}}
}

func TestRemoveCopyRestoresOnFailedSwitch(t *testing.T) {
	ctx := context.Background()
	loc := cluster.Location{Path: t.TempDir()}
	target := writeCopy(t, loc, map[string]string{"model.gguf": "data"})
	file := filepath.Join(target.dir, "model.gguf")

	// Ollama could not replace the link
	m := fakeOllama(t, cluster.RelinkResponse{Changes: []cluster.RelinkResult{{From: file, Error: "permission denied"}}}, func() []cluster.Link { return nil })
	if err := removeCopy(ctx, m, target.dir, false); err == nil || !strings.Contains(err.Error(), "nothing was deleted") {
		t.Fatalf("removeCopy = %v, want a failure", err)
	}
	if _, err := os.Stat(file); err != nil {
		t.Fatalf("the copy must be moved back: %v", err)
	}

	// relink reported nothing, but a link still points into the copy
	m = fakeOllama(t, cluster.RelinkResponse{}, func() []cluster.Link {
		return []cluster.Link{{Digest: "sha256:" + sum("data"), Target: file}}
	})
	if err := removeCopy(ctx, m, target.dir, false); err == nil {
		t.Fatal("removeCopy must fail while a link points into the copy")
	}
	if _, err := os.Stat(file); err != nil {
		t.Fatalf("the copy must be moved back: %v", err)
	}

	// a problem in another repository does not block the deletion
	m = fakeOllama(t, cluster.RelinkResponse{Changes: []cluster.RelinkResult{{From: "/elsewhere/model.gguf", Missing: true}}}, func() []cluster.Link { return nil })
	if err := removeCopy(ctx, m, target.dir, false); err != nil {
		t.Fatalf("removeCopy: %v", err)
	}
	if _, err := os.Stat(target.dir); !os.IsNotExist(err) {
		t.Fatalf("the copy must be deleted: %v", err)
	}
	if _, err := os.Stat(target.dir + ".dropping"); !os.IsNotExist(err) {
		t.Fatalf("nothing may be left aside: %v", err)
	}
}
