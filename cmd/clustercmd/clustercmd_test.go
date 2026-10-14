package clustercmd

import (
	"encoding/json"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func iface(name, cidr, rdma string) netIf {
	ip, n, err := net.ParseCIDR(cidr)
	if err != nil {
		panic(err)
	}
	return netIf{Name: name, Addr: ip, Net: n, RDMADev: rdma}
}

func TestFindLink(t *testing.T) {
	local := []netIf{
		iface("enP7s7", "192.168.5.60/24", ""),
		iface("enp1s0f0np0", "10.100.48.1/24", "rocep1s0f0"),
		iface("enP2p1s0f0np0", "10.100.49.1/24", "roceP2p1s0f0"),
	}
	remote := []netIf{
		iface("enP7s7", "192.168.5.61/24", ""),
		iface("enp1s0f0np0", "10.100.48.2/24", "rocep1s0f0"),
		iface("enP2p1s0f0np0", "10.100.49.2/24", "roceP2p1s0f0"),
	}

	l, ok := findLink(local, remote, "", false)
	if !ok || l.Remote.Addr.String() != "10.100.48.2" || l.Local.Addr.String() != "10.100.48.1" {
		t.Fatalf("auto: %+v %v", l, ok)
	}
	l, ok = findLink(local, remote, "10.100.49.2", false)
	if !ok || l.Local.Addr.String() != "10.100.49.1" {
		t.Fatalf("wanted address: %+v %v", l, ok)
	}
	if _, ok := findLink(local, remote, "192.168.5.61", false); ok {
		t.Fatal("LAN address must be refused without allowLAN")
	}
	if l, ok := findLink(local, remote, "192.168.5.61", true); !ok || l.Local.Addr.String() != "192.168.5.60" {
		t.Fatalf("allowLAN: %+v %v", l, ok)
	}
	if _, ok := findLink(local[:1], remote, "", false); ok {
		t.Fatal("no cable must be refused")
	}
}

func TestParseRepoTag(t *testing.T) {
	for _, in := range []string{"unsloth/Qwen3-4B-GGUF:Q4_K_M", "hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M", "https://huggingface.co/unsloth/Qwen3-4B-GGUF:Q4_K_M"} {
		repo, tag, err := parseRepoTag(in)
		if err != nil || repo != "unsloth/Qwen3-4B-GGUF" || tag != "Q4_K_M" {
			t.Fatalf("%s: %s %s %v", in, repo, tag, err)
		}
	}
	if _, _, err := parseRepoTag("unsloth/Qwen3-4B-GGUF"); err == nil {
		t.Fatal("missing tag must fail")
	}
}

func TestSelectFiles(t *testing.T) {
	files := []hfFile{
		{Path: "Qwen-Q4_K_M.gguf", Size: 10, SHA: "a"},
		{Path: "Qwen-UD-Q4_K_M.gguf", Size: 11, SHA: "b"},
		{Path: "UD-IQ3_XXS/Qwen-UD-IQ3_XXS-00002-of-00002.gguf", Size: 5, SHA: "c2"},
		{Path: "UD-IQ3_XXS/Qwen-UD-IQ3_XXS-00001-of-00002.gguf", Size: 5, SHA: "c1"},
		{Path: "mmproj-BF16.gguf", Size: 3, SHA: "p1"},
		{Path: "mmproj-F16.gguf", Size: 3, SHA: "p2"},
		{Path: "config.json", Size: 1},
		{Path: "README.md", Size: 1},
	}

	model, proj, extra, err := selectFiles(files, "q4_k_m")
	if err != nil || len(model) != 1 || model[0].SHA != "a" {
		t.Fatalf("Q4_K_M must prefer the plain file: %+v %v", model, err)
	}
	if proj == nil || proj.SHA != "p2" {
		t.Fatalf("projector must prefer F16: %+v", proj)
	}
	if len(extra) != 2 {
		t.Fatalf("extra files: %+v", extra)
	}

	model, _, _, err = selectFiles(files, "UD-Q4_K_M")
	if err != nil || len(model) != 1 || model[0].SHA != "b" {
		t.Fatalf("UD-Q4_K_M: %+v %v", model, err)
	}

	model, _, _, err = selectFiles(files, "IQ3_XXS")
	if err != nil || len(model) != 2 || model[0].SHA != "c1" {
		t.Fatalf("sharded IQ3_XXS in order: %+v %v", model, err)
	}

	if _, _, _, err := selectFiles(files, "Q8_0"); err == nil {
		t.Fatal("unknown tag must fail")
	}
}

func TestImportFollowsLinkChain(t *testing.T) {
	cache, loc := t.TempDir(), t.TempDir()
	sha := strings.Repeat("ab", 32)
	repoDir := filepath.Join(cache, "models--owner--repo")
	shared := filepath.Join(cache, "blobs", "5e", "data")
	mustMkdir(t, filepath.Dir(shared), filepath.Join(repoDir, "blobs"), filepath.Join(repoDir, "refs"), filepath.Join(repoDir, "snapshots", "rev1", "Q4"))
	if err := os.WriteFile(shared, []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("../../blobs/5e/data", filepath.Join(repoDir, "blobs", sha)); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("../../../blobs/"+sha, filepath.Join(repoDir, "snapshots", "rev1", "Q4", "m.gguf")); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repoDir, "refs", "main"), []byte("rev1"), 0o644); err != nil {
		t.Fatal(err)
	}

	repos, err := scanHFCache(cache, loc, nil)
	if err != nil || len(repos) != 1 || len(repos[0].files) != 1 {
		t.Fatalf("scan: %+v %v", repos, err)
	}
	f := repos[0].files[0]
	if f.sha != sha || f.blob != shared || f.done || f.path != "Q4/m.gguf" {
		t.Fatalf("file: %+v", f)
	}

	// move like runImport does, then the scan reports it as done and the
	// cache path still reads the data
	mustMkdir(t, filepath.Dir(f.dest))
	if err := os.Rename(f.blob, f.dest); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(f.dest, f.blob); err != nil {
		t.Fatal(err)
	}
	repos, _ = scanHFCache(cache, loc, nil)
	if !repos[0].files[0].done {
		t.Fatal("moved file must be reported as done")
	}
	data, err := os.ReadFile(filepath.Join(repoDir, "snapshots", "rev1", "Q4", "m.gguf"))
	if err != nil || string(data) != "GGUF" {
		t.Fatalf("cache path broken: %q %v", data, err)
	}
}

func mustMkdir(t *testing.T, dirs ...string) {
	t.Helper()
	for _, d := range dirs {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
}

func TestHostListenAddr(t *testing.T) {
	for _, c := range []struct {
		ports []string
		want  string
	}{
		{nil, ""},
		{[]string{"11434:11434/tcp"}, ""},
		{[]string{"0.0.0.0:11434:11434/tcp", ":::11434:11434/tcp"}, ""},
		{[]string{"8080:11434/tcp"}, "0.0.0.0:8080"},
		{[]string{"127.0.0.1:11434:11434/tcp"}, "127.0.0.1:11434"},
		{[]string{"192.168.5.60:11500:11434/tcp"}, "192.168.5.60:11500"},
		{[]string{"127.0.0.1:11434:11434/tcp", "0.0.0.0:11434:11434/tcp"}, ""},
		{[]string{"::1:11434:11434/tcp", "127.0.0.1:11434:11434/tcp"}, "127.0.0.1:11434"},
		{[]string{"::1:11434:11434/tcp"}, "[::1]:11434"},
		{[]string{"127.0.0.1:3000:8080/tcp"}, ""},
	} {
		if got := hostListenAddr(c.ports); got != c.want {
			t.Errorf("%v: got %q, want %q", c.ports, got, c.want)
		}
	}
}

func TestSpecKeepsSettings(t *testing.T) {
	var c containerInfo
	inspect := `{"Name":"/ollama","Config":{"Image":"ollama/ollama","User":"1000:1000","Labels":{"com.example":"x","org.opencontainers.image.version":"24.04"}},
		"HostConfig":{"NetworkMode":"host","Mounts":[{"Type":"bind","Source":"/data/models","Target":"/root/.ollama","ReadOnly":false},{"Type":"volume","Source":"cache","Target":"/cache","ReadOnly":true}],
		"Memory":8589934592,"NanoCpus":2500000000,"Tmpfs":{"/tmp":"size=1g"},"SecurityOpt":["no-new-privileges"],"LogConfig":{"Type":"local","Config":{"max-size":"10m"}}}}`
	if err := json.Unmarshal([]byte(inspect), &c); err != nil {
		t.Fatal(err)
	}
	args := strings.Join(specFromContainer(t.Context(), &c).runArgs(), " ")
	for _, want := range []string{
		"--mount type=bind,target=/root/.ollama,source=/data/models",
		"--mount type=volume,target=/cache,source=cache,readonly",
		"--user 1000:1000", "--label com.example=x", "--memory 8589934592", "--cpus 2.5",
		"--tmpfs /tmp:size=1g", "--security-opt no-new-privileges", "--log-driver local --log-opt max-size=10m",
	} {
		if !strings.Contains(args, want) {
			t.Errorf("missing %q in %s", want, args)
		}
	}
	if strings.Contains(args, "org.opencontainers") {
		t.Errorf("image label carried over: %s", args)
	}
}
