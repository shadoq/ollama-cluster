package cluster

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestMoveBlob(t *testing.T) {
	for _, copied := range []bool{false, true} {
		name := "hard link"
		if copied {
			name = "copy"
		}
		t.Run(name, func(t *testing.T) {
			if copied {
				linkFile = func(string, string) error { return errors.New("another disk") }
				t.Cleanup(func() { linkFile = os.Link })
			}
			models := t.TempDir()
			t.Setenv("OLLAMA_MODELS", models)
			if err := os.MkdirAll(filepath.Join(models, "blobs"), 0o755); err != nil {
				t.Fatal(err)
			}
			a, b, archive := t.TempDir(), t.TempDir(), t.TempDir()
			cfg := &Config{Locations: []Location{{Path: a}, {Path: b}, {Path: archive, Archive: true}}}

			digest := digestOf("weights")
			blob, _ := blobPath(digest)
			if err := os.WriteFile(blob, []byte("weights"), 0o644); err != nil {
				t.Fatal(err)
			}
			ctx := context.Background()

			if _, err := MoveBlob(ctx, cfg, digest, archive, nil); err == nil {
				t.Fatal("a move to an archive location must fail")
			}
			if _, err := MoveBlob(ctx, cfg, digest, t.TempDir(), nil); err == nil {
				t.Fatal("a move outside the locations must fail")
			}

			// store -> a: the blob becomes a link, the data lives in a
			res, err := MoveBlob(ctx, cfg, digest, a, nil)
			if err != nil || res.To != MovedPath(a, digest) || res.Copied != copied {
				t.Fatalf("move to a: %+v %v", res, err)
			}
			checkLink(t, blob, MovedPath(a, digest))
			if res, err := MoveBlob(ctx, cfg, digest, a, nil); err != nil || res.Skipped == "" {
				t.Fatalf("a second move to a must be a no-op: %+v %v", res, err)
			}

			// a -> b: the file in a is removed
			if _, err := MoveBlob(ctx, cfg, digest, b, nil); err != nil {
				t.Fatal(err)
			}
			checkLink(t, blob, MovedPath(b, digest))
			if _, err := os.Stat(MovedPath(a, digest)); !os.IsNotExist(err) {
				t.Fatalf("the old file in a must be gone: %v", err)
			}
			if changes, err := Relink(cfg); err != nil || len(changes) != 0 {
				t.Fatalf("Relink after a move must change nothing: %+v %v", changes, err)
			}

			// b -> store: a regular blob again
			if _, err := MoveBlob(ctx, cfg, digest, "", nil); err != nil {
				t.Fatal(err)
			}
			fi, err := os.Lstat(blob)
			if err != nil || !fi.Mode().IsRegular() {
				t.Fatalf("the blob must be a regular file again: %v %v", fi, err)
			}
			if data, _ := os.ReadFile(blob); string(data) != "weights" {
				t.Fatalf("blob content %q", data)
			}
			if _, err := os.Stat(MovedPath(b, digest)); !os.IsNotExist(err) {
				t.Fatalf("the file in b must be gone: %v", err)
			}
			if res, err := MoveBlob(ctx, cfg, digest, "", nil); err != nil || res.Skipped == "" {
				t.Fatalf("a second move to the store must be a no-op: %+v %v", res, err)
			}
		})
	}
}

func TestMoveBlobChecksContent(t *testing.T) {
	linkFile = func(string, string) error { return errors.New("another disk") }
	t.Cleanup(func() { linkFile = os.Link })
	models := t.TempDir()
	t.Setenv("OLLAMA_MODELS", models)
	if err := os.MkdirAll(filepath.Join(models, "blobs"), 0o755); err != nil {
		t.Fatal(err)
	}
	a := t.TempDir()
	cfg := &Config{Locations: []Location{{Path: a}}}

	// a damaged blob is never moved: the store keeps it as it is
	digest := digestOf("weights")
	blob, _ := blobPath(digest)
	if err := os.WriteFile(blob, []byte("damaged"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := MoveBlob(context.Background(), cfg, digest, a, nil); err == nil {
		t.Fatal("a blob with other content must not move")
	}
	if fi, err := os.Lstat(blob); err != nil || !fi.Mode().IsRegular() {
		t.Fatalf("the blob must stay in the store: %v %v", fi, err)
	}
	if entries, _ := os.ReadDir(filepath.Join(a, MovedDir)); len(entries) != 0 {
		t.Fatalf("nothing may be left in the location: %v", entries)
	}

	// an interrupted copy changes nothing
	if err := os.WriteFile(blob, []byte("weights"), 0o644); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := MoveBlob(ctx, cfg, digest, a, nil); err == nil {
		t.Fatal("a cancelled move must fail")
	}
	if fi, err := os.Lstat(blob); err != nil || !fi.Mode().IsRegular() {
		t.Fatalf("the blob must stay in the store: %v %v", fi, err)
	}
}

func TestMoveBlobKeepsRepositoryCopies(t *testing.T) {
	models := t.TempDir()
	t.Setenv("OLLAMA_MODELS", models)
	if err := os.MkdirAll(filepath.Join(models, "blobs"), 0o755); err != nil {
		t.Fatal(err)
	}
	a, b := t.TempDir(), t.TempDir()
	cfg := &Config{Locations: []Location{{Path: a}, {Path: b}}}
	file := filepath.Join(a, HFDir, "owner", "repo", "model.gguf")
	if err := os.MkdirAll(filepath.Dir(file), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(file, []byte("gguf"), 0o644); err != nil {
		t.Fatal(err)
	}
	digest := digestOf("gguf")
	if _, err := LinkBlob(cfg, digest, file); err != nil {
		t.Fatal(err)
	}
	res, err := MoveBlob(context.Background(), cfg, digest, b, nil)
	if err != nil || res.Skipped == "" {
		t.Fatalf("a file of a repository copy must stay: %+v %v", res, err)
	}
	blob, _ := blobPath(digest)
	checkLink(t, blob, file)
}

func checkLink(t *testing.T, blob, want string) {
	t.Helper()
	target, err := os.Readlink(blob)
	if err != nil || target != want {
		t.Fatalf("blob points at %q (%v), want %q", target, err, want)
	}
	if data, err := os.ReadFile(blob); err != nil || len(data) == 0 {
		t.Fatalf("blob unreadable through the link: %v", err)
	}
}
