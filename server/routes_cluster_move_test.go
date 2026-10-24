package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cluster"
	"github.com/ollama/ollama/fs/gguf"
	gguftest "github.com/ollama/ollama/internal/testutil/gguf"
	"github.com/ollama/ollama/manifest"
)

func TestClusterMove(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	loc := t.TempDir()
	if err := (&cluster.Config{Locations: []cluster.Location{{Path: loc}}}).Save(); err != nil {
		t.Fatal(err)
	}

	var s Server
	n := uint64(cluster.MinMoveSize / 4)
	_, digest := createBinFile(t, nil, []*gguftest.Tensor{
		{Name: "weights", Type: gguf.TensorTypeF32, Shape: []uint64{n}, WriterTo: bytes.NewReader(make([]byte, 4*n))},
	})
	if w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Name:     "test",
		Files:    map[string]string{"test.gguf": digest},
		Template: "{{ .Prompt }}",
	}); w.Code != http.StatusOK {
		t.Fatalf("create: %d %s", w.Code, w.Body)
	}

	r := gin.New()
	r.POST("/api/cluster/move", s.ClusterMoveHandler)
	move := func(to string) []cluster.MoveProgress {
		t.Helper()
		body, _ := json.Marshal(cluster.MoveRequest{Model: "test", To: to})
		w := NewRecorder()
		r.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/cluster/move", bytes.NewReader(body)))
		var events []cluster.MoveProgress
		for line := range strings.SplitSeq(strings.TrimSpace(w.Body.String()), "\n") {
			if strings.Contains(line, `"error"`) {
				t.Fatalf("move to %q: %s", to, line)
			}
			var ev cluster.MoveProgress
			if err := json.Unmarshal([]byte(line), &ev); err != nil {
				t.Fatalf("%q: %v", line, err)
			}
			events = append(events, ev)
		}
		if len(events) == 0 || events[len(events)-1].Status != "success" {
			t.Fatalf("move to %q: no success: %+v", to, events)
		}
		return events
	}

	blob, _ := manifest.BlobsPath(digest)
	events := move(loc)
	moved := 0
	for _, ev := range events {
		if ev.Status == "moved" {
			moved++
			if ev.Digest != digest {
				t.Fatalf("only the weights may move, moved %s", ev.Digest)
			}
		}
	}
	if moved != 1 {
		t.Fatalf("moved %d blobs, want 1: %+v", moved, events)
	}
	if target, err := os.Readlink(blob); err != nil || target != cluster.MovedPath(loc, digest) {
		t.Fatalf("blob points at %q (%v)", target, err)
	}

	// the model still shows and the store keeps the small blobs
	if w := createRequest(t, s.ShowHandler, api.ShowRequest{Model: "test"}); w.Code != http.StatusOK {
		t.Fatalf("show after the move: %d %s", w.Code, w.Body)
	}

	move("")
	if fi, err := os.Lstat(blob); err != nil || !fi.Mode().IsRegular() {
		t.Fatalf("the blob must be back in the store: %v %v", fi, err)
	}
	if _, err := os.Stat(cluster.MovedPath(loc, digest)); !os.IsNotExist(err) {
		t.Fatalf("the moved file must be gone: %v", err)
	}
}
