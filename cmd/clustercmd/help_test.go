package clustercmd

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/cluster"
)

func TestEveryCommandHasHelp(t *testing.T) {
	root := Command()
	walk(root, func(c *cobra.Command) {
		if c == root || c.Name() == "help" {
			return
		}
		if c.Long == "" || c.Example == "" {
			t.Errorf("%s: no long help or example", c.CommandPath())
		}
		if c.Parent() == root && c.GroupID == "" {
			t.Errorf("%s: no group", c.CommandPath())
		}
	})
	for key := range helpTexts {
		if c, rest, err := root.Find(strings.Fields(key)); err != nil || len(rest) > 0 || c == root {
			t.Errorf("help text for unknown command %q", key)
		}
	}
}

func TestHelpEverywhere(t *testing.T) {
	cases := map[string]string{
		"locations help":     "--archive",
		"help locations":     "--archive",
		"locations add help": "--position",
		"help locations add": "--position",
		"help pull":          "UD-IQ3_XXS",
		"pull help":          "UD-IQ3_XXS",
		"drop help":          "--force",
		"ls help":            "CAPABILITIES",
		"ps --help":          "UNLOADS",
		"help":               "Model files and copies from Hugging Face:",
		"mv help":            "ollama/blobs",
	}
	for args, want := range cases {
		root := Command()
		var out bytes.Buffer
		root.SetOut(&out)
		root.SetErr(&out)
		root.SetArgs(strings.Fields(args))
		if err := root.Execute(); err != nil {
			t.Errorf("%s: %v", args, err)
		}
		if !strings.Contains(out.String(), want) {
			t.Errorf("%s: output has no %q:\n%s", args, want, out.String())
		}
	}
}

func TestFits(t *testing.T) {
	const gb = 1 << 30
	for _, c := range []struct {
		size          int64
		local, remote uint64
		want          string
	}{
		{81 * gb, 120 * gb, 120 * gb, "here"},
		{141 * gb, 120 * gb, 120 * gb, "split"},
		{141 * gb, 120 * gb, 0, "too large"},
		{300 * gb, 120 * gb, 120 * gb, "too large"},
		{5 * gb, 0, 0, "here"},
	} {
		if got := fits(c.size, c.local, c.remote); got != c.want {
			t.Errorf("fits(%d GB, %d, %d) = %s, want %s", c.size/gb, c.local/gb, c.remote/gb, got, c.want)
		}
	}
}

func TestWhereFiles(t *testing.T) {
	cfg := &cluster.Config{Locations: []cluster.Location{{Path: "/mnt/a"}, {Path: "/mnt/b"}}}
	big := int64(dataBlob)

	where, repo, broken := whereFiles(cfg, []cluster.BlobFile{{Digest: "sha256:1", Size: big, OK: true}})
	if where != "store" || repo != "" || broken != 0 {
		t.Errorf("store: %s %s %d", where, repo, broken)
	}

	bs := []cluster.BlobFile{
		{Digest: "sha256:1", Size: big, Target: "/mnt/a/hf/unsloth/Qwen3-4B-GGUF/Q4/x-00001-of-00002.gguf", OK: true},
		{Digest: "sha256:2", Size: big, Target: "/mnt/a/hf/unsloth/Qwen3-4B-GGUF/Q4/x-00002-of-00002.gguf", OK: false},
	}
	where, repo, broken = whereFiles(cfg, bs)
	if where != "/mnt/a" || repo != "unsloth/Qwen3-4B-GGUF" || broken != 1 {
		t.Errorf("copy: %s %s %d", where, repo, broken)
	}

	bs[1].Target = "/mnt/b/hf/unsloth/Qwen3-4B-GGUF/Q4/x-00002-of-00002.gguf"
	if where, _, _ = whereFiles(cfg, bs); where != "mixed" {
		t.Errorf("mixed: %s", where)
	}

	// metadata blobs do not count, a broken link always does
	small := dataBlobs([]cluster.BlobFile{{Size: 100, OK: true}, {Size: 0, Target: "/gone", OK: false}, {Size: big, OK: true}})
	if len(small) != 2 {
		t.Errorf("dataBlobs: %+v", small)
	}
}

func TestPSColumns(t *testing.T) {
	st := &cluster.StatusResponse{
		Workers: []cluster.WorkerCheck{{Worker: cluster.Worker{Name: "gx10-bf3f", Addr: "10.100.48.2"}}},
	}
	l := cluster.LoadedModel{
		Context: 32768, TrainContext: 262144,
		Devices: []cluster.DeviceMemory{
			{Name: "CUDA0", Size: 64_600_000_000},
			{Name: "RPC0[10.100.48.2:50052]", Remote: true, Size: 77_800_000_000},
		},
		ExpiresAt: time.Now().Add(4*time.Minute + 30*time.Second),
	}
	if got := placement(st, l); got != "split: here 64.6 GB + gx10-bf3f 77.8 GB" {
		t.Errorf("placement: %s", got)
	}
	if got := loadedContext(l); got != "32k of 256k" {
		t.Errorf("context: %s", got)
	}
	if got := unloads(l); !strings.HasPrefix(got, "in 4 min") {
		t.Errorf("unloads: %s", got)
	}
	l.Active = 2
	if got := unloads(l); got != "in use" {
		t.Errorf("unloads active: %s", got)
	}
	l.Active, l.KeepAlive = 0, -1
	if got := unloads(l); got != "never" {
		t.Errorf("unloads forever: %s", got)
	}
}

func TestBenchModes(t *testing.T) {
	for in, want := range map[string]string{"": "here,split,worker,tensor", "worker,local": "here,worker", "tensor,here": "here,tensor", "split": "split", "split,split": "split"} {
		got, err := benchModes(in)
		if err != nil || strings.Join(got, ",") != want {
			t.Errorf("%q: %v %v", in, got, err)
		}
	}
	if _, err := benchModes("fast"); err == nil {
		t.Error("unknown mode accepted")
	}
}
