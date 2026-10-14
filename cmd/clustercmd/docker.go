package clustercmd

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/envconfig"
)

// containerInfo is the part of `docker inspect` needed to recreate a container.
type containerInfo struct {
	ID     string `json:"Id"`
	Name   string `json:"Name"`
	Image  string `json:"Image"` // image ID
	Config struct {
		Image      string            `json:"Image"`
		Env        []string          `json:"Env"`
		Cmd        []string          `json:"Cmd"`
		Entrypoint []string          `json:"Entrypoint"`
		Labels     map[string]string `json:"Labels"`
		User       string            `json:"User"`
		WorkingDir string            `json:"WorkingDir"`
	} `json:"Config"`
	HostConfig struct {
		Binds        []string `json:"Binds"`
		NetworkMode  string   `json:"NetworkMode"`
		PortBindings map[string][]struct {
			HostIP   string `json:"HostIp"`
			HostPort string `json:"HostPort"`
		} `json:"PortBindings"`
		RestartPolicy struct {
			Name              string `json:"Name"`
			MaximumRetryCount int    `json:"MaximumRetryCount"`
		} `json:"RestartPolicy"`
		DeviceRequests []struct {
			Driver       string     `json:"Driver"`
			Count        int        `json:"Count"`
			DeviceIDs    []string   `json:"DeviceIDs"`
			Capabilities [][]string `json:"Capabilities"`
		} `json:"DeviceRequests"`
		Devices []deviceMapping `json:"Devices"`
		Ulimits []struct {
			Name string `json:"Name"`
			Soft int64  `json:"Soft"`
			Hard int64  `json:"Hard"`
		} `json:"Ulimits"`
		Runtime    string   `json:"Runtime"`
		ShmSize    int64    `json:"ShmSize"`
		Privileged bool     `json:"Privileged"`
		CapAdd     []string `json:"CapAdd"`
		ExtraHosts []string `json:"ExtraHosts"`
		IpcMode    string   `json:"IpcMode"`

		// carried over as they are, so recreating keeps them
		Mounts []struct {
			Type     string `json:"Type"`
			Source   string `json:"Source"`
			Target   string `json:"Target"`
			ReadOnly bool   `json:"ReadOnly"`
		} `json:"Mounts"`
		CapDrop     []string          `json:"CapDrop"`
		SecurityOpt []string          `json:"SecurityOpt"`
		GroupAdd    []string          `json:"GroupAdd"`
		Tmpfs       map[string]string `json:"Tmpfs"`
		Memory      int64             `json:"Memory"`
		NanoCpus    int64             `json:"NanoCpus"`
		PidMode     string            `json:"PidMode"`
		LogConfig   struct {
			Type   string            `json:"Type"`
			Config map[string]string `json:"Config"`
		} `json:"LogConfig"`
	} `json:"HostConfig"`
	Mounts []struct {
		Type        string `json:"Type"`
		Source      string `json:"Source"`
		Destination string `json:"Destination"`
		RW          bool   `json:"RW"`
	} `json:"Mounts"`
	State struct {
		Running    bool   `json:"Running"`
		Status     string `json:"Status"`
		Restarting bool   `json:"Restarting"`
	} `json:"State"`
}

func (c *containerInfo) name() string {
	return strings.TrimPrefix(c.Name, "/")
}

func (c *containerInfo) env(key string) (string, bool) {
	for _, kv := range c.Config.Env {
		if k, v, ok := strings.Cut(kv, "="); ok && k == key {
			return v, true
		}
	}
	return "", false
}

type deviceMapping struct {
	PathOnHost        string `json:"PathOnHost"`
	PathInContainer   string `json:"PathInContainer"`
	CgroupPermissions string `json:"CgroupPermissions"`
}

func (c *containerInfo) hasDevice(path string) bool {
	return slices.ContainsFunc(c.HostConfig.Devices, func(d deviceMapping) bool { return d.PathOnHost == path })
}

func (c *containerInfo) memlockUnlimited() bool {
	for _, u := range c.HostConfig.Ulimits {
		if u.Name == "memlock" && u.Soft == -1 && u.Hard == -1 {
			return true
		}
	}
	return false
}

// hostPath maps a path inside the container to the host through bind mounts.
func (c *containerInfo) hostPath(inContainer string) (string, bool) {
	best := ""
	host := ""
	for _, m := range c.Mounts {
		if m.Type != "bind" {
			continue
		}
		rel, err := filepath.Rel(m.Destination, inContainer)
		if err != nil || strings.HasPrefix(rel, "..") {
			continue
		}
		if len(m.Destination) > len(best) {
			best = m.Destination
			host = filepath.Join(m.Source, rel)
		}
	}
	return host, best != ""
}

// mounted reports whether a host path is visible in the container at the
// same path.
func (c *containerInfo) mounted(path string) bool {
	for _, m := range c.Mounts {
		if m.Type != "bind" {
			continue
		}
		rel, err := filepath.Rel(m.Destination, path)
		if err != nil || strings.HasPrefix(rel, "..") {
			continue
		}
		hostRel, err := filepath.Rel(m.Source, path)
		if err == nil && hostRel == rel {
			return true
		}
	}
	return false
}

func docker(ctx context.Context, args ...string) (string, error) {
	cmd := exec.CommandContext(ctx, "docker", args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = err.Error()
		}
		return string(out), fmt.Errorf("docker %s: %s", args[0], msg)
	}
	return strings.TrimSpace(string(out)), nil
}

func dockerAvailable(ctx context.Context) error {
	_, err := docker(ctx, "info", "--format", "{{.ServerVersion}}")
	return err
}

func inspectContainer(ctx context.Context, name string) (*containerInfo, error) {
	out, err := docker(ctx, "inspect", "--type", "container", name)
	if err != nil {
		return nil, err
	}
	var infos []containerInfo
	if err := json.Unmarshal([]byte(out), &infos); err != nil {
		return nil, err
	}
	if len(infos) == 0 {
		return nil, fmt.Errorf("container %s not found", name)
	}
	return &infos[0], nil
}

// findOllamaContainer returns the running container that serves the Ollama
// API this command talks to.
func findOllamaContainer(ctx context.Context, name string) (*containerInfo, error) {
	if name != "" {
		return inspectContainer(ctx, name)
	}
	if err := dockerAvailable(ctx); err != nil {
		return nil, nil //nolint:nilerr // no Docker means native Ollama
	}

	out, err := docker(ctx, "ps", "--format", "{{.Names}}")
	if err != nil {
		return nil, err
	}
	port := envconfig.Host().Port()

	var candidates []*containerInfo
	for _, n := range strings.Fields(out) {
		c, err := inspectContainer(ctx, n)
		if err != nil || !isOllamaServe(c) {
			continue
		}
		candidates = append(candidates, c)
	}

	var matching []*containerInfo
	for _, c := range candidates {
		if containerServesPort(c, port) {
			matching = append(matching, c)
		}
	}
	switch {
	case len(matching) == 1:
		return matching[0], nil
	case len(matching) > 1:
		names := make([]string, len(matching))
		for i, c := range matching {
			names[i] = c.name()
		}
		return nil, fmt.Errorf("several Ollama containers serve port %s (%s); choose one with --container", port, strings.Join(names, ", "))
	}
	return nil, nil
}

func isOllamaServe(c *containerInfo) bool {
	entry := strings.Join(c.Config.Entrypoint, " ")
	cmd := strings.Join(c.Config.Cmd, " ")
	return strings.Contains(entry, "ollama") && strings.Contains(cmd, "serve")
}

func containerServesPort(c *containerInfo, port string) bool {
	if c.HostConfig.NetworkMode == "host" {
		host, _ := c.env("OLLAMA_HOST")
		if host == "" {
			return port == "11434"
		}
		return strings.HasSuffix(host, ":"+port)
	}
	for _, bindings := range c.HostConfig.PortBindings {
		for _, b := range bindings {
			if b.HostPort == port {
				return true
			}
		}
	}
	return false
}

// imageHasCluster reports whether an image ships the RPC worker.
func imageHasCluster(ctx context.Context, image string) bool {
	_, err := docker(ctx, "run", "--rm", "--entrypoint", "/bin/sh", image, "-c", "test -x /usr/lib/ollama/ggml-rpc-server")
	return err == nil
}

func imageID(ctx context.Context, image string) (string, error) {
	return docker(ctx, "image", "inspect", "--format", "{{.Id}}", image)
}

func imageEnv(ctx context.Context, image string) []string {
	out, err := docker(ctx, "image", "inspect", "--format", "{{json .Config.Env}}", image)
	if err != nil {
		return nil
	}
	var env []string
	_ = json.Unmarshal([]byte(out), &env)
	return env
}

// containerSpec is what `docker run` needs to create the main container.
type containerSpec struct {
	Name          string
	Image         string
	Env           []string
	Cmd           []string
	Entrypoint    []string
	Binds         []string
	NetworkMode   string
	Ports         []string // -p values
	Restart       string
	GPUs          string // --gpus value
	Devices       []string
	Ulimits       []string
	Runtime       string
	ShmSize       int64
	Privileged    bool
	CapAdd        []string
	ExtraHosts    []string
	IpcMode       string
	DefaultCmdEnv []string // env of the image, not repeated on the command line

	Mounts      []string // --mount values
	User        string
	WorkingDir  string
	Labels      []string // --label values, sorted
	CapDrop     []string
	SecurityOpt []string
	GroupAdd    []string
	Tmpfs       []string // --tmpfs values, sorted
	Memory      int64
	NanoCpus    int64
	PidMode     string
	LogDriver   string
	LogOpts     []string // --log-opt values, sorted
}

// specFromContainer captures a container so it can be created again.
func specFromContainer(ctx context.Context, c *containerInfo) containerSpec {
	s := containerSpec{
		Name:        c.name(),
		Image:       c.Config.Image,
		Cmd:         slices.Clone(c.Config.Cmd),
		Entrypoint:  slices.Clone(c.Config.Entrypoint),
		Binds:       slices.Clone(c.HostConfig.Binds),
		NetworkMode: c.HostConfig.NetworkMode,
		Runtime:     c.HostConfig.Runtime,
		ShmSize:     c.HostConfig.ShmSize,
		Privileged:  c.HostConfig.Privileged,
		CapAdd:      slices.Clone(c.HostConfig.CapAdd),
		ExtraHosts:  slices.Clone(c.HostConfig.ExtraHosts),
		IpcMode:     c.HostConfig.IpcMode,
	}

	// Keep only the variables set for this container, not the image defaults,
	// so a new image brings its own defaults.
	defaults := imageEnv(ctx, c.Image)
	for _, kv := range c.Config.Env {
		if !slices.Contains(defaults, kv) {
			s.Env = append(s.Env, kv)
		}
	}

	for port, bindings := range c.HostConfig.PortBindings {
		for _, b := range bindings {
			v := b.HostPort + ":" + port
			if b.HostIP != "" {
				v = b.HostIP + ":" + v
			}
			s.Ports = append(s.Ports, v)
		}
	}
	slices.Sort(s.Ports)

	if r := c.HostConfig.RestartPolicy; r.Name != "" && r.Name != "no" {
		s.Restart = r.Name
		if r.Name == "on-failure" && r.MaximumRetryCount > 0 {
			s.Restart += ":" + strconv.Itoa(r.MaximumRetryCount)
		}
	}
	for _, d := range c.HostConfig.DeviceRequests {
		if slices.ContainsFunc(d.Capabilities, func(caps []string) bool { return slices.Contains(caps, "gpu") }) {
			if len(d.DeviceIDs) > 0 {
				s.GPUs = "device=" + strings.Join(d.DeviceIDs, ",")
			} else if d.Count < 0 {
				s.GPUs = "all"
			} else {
				s.GPUs = strconv.Itoa(d.Count)
			}
		}
	}
	for _, d := range c.HostConfig.Devices {
		v := d.PathOnHost
		if d.PathInContainer != "" && d.PathInContainer != d.PathOnHost {
			v += ":" + d.PathInContainer
		}
		s.Devices = append(s.Devices, v)
	}
	for _, u := range c.HostConfig.Ulimits {
		s.Ulimits = append(s.Ulimits, fmt.Sprintf("%s=%d:%d", u.Name, u.Soft, u.Hard))
	}

	for _, m := range c.HostConfig.Mounts {
		v := "type=" + m.Type + ",target=" + m.Target
		if m.Source != "" {
			v += ",source=" + m.Source
		}
		if m.ReadOnly {
			v += ",readonly"
		}
		s.Mounts = append(s.Mounts, v)
	}
	s.User, s.WorkingDir = c.Config.User, c.Config.WorkingDir
	for k, v := range c.Config.Labels {
		// labels of the image come with the new image
		if !strings.HasPrefix(k, "org.opencontainers.image.") {
			s.Labels = append(s.Labels, k+"="+v)
		}
	}
	slices.Sort(s.Labels)
	s.CapDrop = slices.Clone(c.HostConfig.CapDrop)
	s.SecurityOpt = slices.Clone(c.HostConfig.SecurityOpt)
	s.GroupAdd = slices.Clone(c.HostConfig.GroupAdd)
	for k, v := range c.HostConfig.Tmpfs {
		if v != "" {
			k += ":" + v
		}
		s.Tmpfs = append(s.Tmpfs, k)
	}
	slices.Sort(s.Tmpfs)
	s.Memory, s.NanoCpus, s.PidMode = c.HostConfig.Memory, c.HostConfig.NanoCpus, c.HostConfig.PidMode
	if lc := c.HostConfig.LogConfig; lc.Type != "" {
		s.LogDriver = lc.Type
		for k, v := range lc.Config {
			s.LogOpts = append(s.LogOpts, k+"="+v)
		}
		slices.Sort(s.LogOpts)
	}
	return s
}

func (s containerSpec) runArgs() []string {
	args := []string{"run", "-d", "--name", s.Name}
	if s.Restart != "" {
		args = append(args, "--restart", s.Restart)
	}
	if s.GPUs != "" {
		args = append(args, "--gpus", s.GPUs)
	}
	if s.NetworkMode != "" && s.NetworkMode != "default" {
		args = append(args, "--network", s.NetworkMode)
	}
	if s.NetworkMode != "host" {
		for _, p := range s.Ports {
			args = append(args, "-p", p)
		}
	}
	for _, b := range s.Binds {
		args = append(args, "-v", b)
	}
	for _, d := range s.Devices {
		args = append(args, "--device", d)
	}
	for _, u := range s.Ulimits {
		args = append(args, "--ulimit", u)
	}
	for _, e := range s.Env {
		args = append(args, "-e", e)
	}
	if s.Runtime != "" && s.Runtime != "runc" {
		args = append(args, "--runtime", s.Runtime)
	}
	if s.ShmSize > 0 && s.ShmSize != 64<<20 {
		args = append(args, "--shm-size", strconv.FormatInt(s.ShmSize, 10))
	}
	if s.Privileged {
		args = append(args, "--privileged")
	}
	for _, c := range s.CapAdd {
		args = append(args, "--cap-add", c)
	}
	for _, h := range s.ExtraHosts {
		args = append(args, "--add-host", h)
	}
	if s.IpcMode != "" && s.IpcMode != "private" && s.IpcMode != "shareable" {
		args = append(args, "--ipc", s.IpcMode)
	}
	for _, m := range s.Mounts {
		args = append(args, "--mount", m)
	}
	if s.User != "" {
		args = append(args, "--user", s.User)
	}
	if s.WorkingDir != "" {
		args = append(args, "--workdir", s.WorkingDir)
	}
	for _, l := range s.Labels {
		args = append(args, "--label", l)
	}
	for _, c := range s.CapDrop {
		args = append(args, "--cap-drop", c)
	}
	for _, o := range s.SecurityOpt {
		args = append(args, "--security-opt", o)
	}
	for _, g := range s.GroupAdd {
		args = append(args, "--group-add", g)
	}
	for _, t := range s.Tmpfs {
		args = append(args, "--tmpfs", t)
	}
	if s.Memory > 0 {
		args = append(args, "--memory", strconv.FormatInt(s.Memory, 10))
	}
	if s.NanoCpus > 0 {
		args = append(args, "--cpus", strconv.FormatFloat(float64(s.NanoCpus)/1e9, 'f', -1, 64))
	}
	if s.PidMode != "" {
		args = append(args, "--pid", s.PidMode)
	}
	if s.LogDriver != "" {
		args = append(args, "--log-driver", s.LogDriver)
		for _, o := range s.LogOpts {
			args = append(args, "--log-opt", o)
		}
	}
	if len(s.Entrypoint) > 0 {
		args = append(args, "--entrypoint", s.Entrypoint[0])
	}
	args = append(args, s.Image)
	if len(s.Entrypoint) > 1 {
		args = append(args, s.Entrypoint[1:]...)
	}
	args = append(args, s.Cmd...)
	return args
}

func (s *containerSpec) setEnv(key, value string) {
	for i, kv := range s.Env {
		if k, _, ok := strings.Cut(kv, "="); ok && k == key {
			s.Env[i] = key + "=" + value
			return
		}
	}
	s.Env = append(s.Env, key+"="+value)
}

func (s *containerSpec) getEnv(key string) (string, bool) {
	for _, kv := range s.Env {
		if k, v, ok := strings.Cut(kv, "="); ok && k == key {
			return v, true
		}
	}
	return "", false
}

func (s *containerSpec) addDevice(path string) {
	if !slices.Contains(s.Devices, path) {
		s.Devices = append(s.Devices, path)
	}
}

func (s *containerSpec) setUlimit(name, value string) {
	s.Ulimits = slices.DeleteFunc(s.Ulimits, func(u string) bool { return strings.HasPrefix(u, name+"=") })
	s.Ulimits = append(s.Ulimits, name+"="+value)
}

func (s *containerSpec) addBind(path string) {
	bind := path + ":" + path
	for _, b := range s.Binds {
		src, dst, _ := strings.Cut(b, ":")
		dst, _, _ = strings.Cut(dst, ":")
		if src == path && dst == path {
			return
		}
	}
	s.Binds = append(s.Binds, bind)
}

// stateDir holds backups of replaced containers on the host.
func stateDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		home = os.TempDir()
	}
	return filepath.Join(home, ".ollama-cluster")
}

// state remembers what `add` changed so `remove` can undo it.
type state struct {
	Container     string    `json:"container,omitempty"`
	Backup        string    `json:"backup,omitempty"`
	OriginalImage string    `json:"original_image,omitempty"`
	ChangedAt     time.Time `json:"changed_at"`
}

func loadState() (*state, error) {
	data, err := os.ReadFile(filepath.Join(stateDir(), "state.json"))
	if errors.Is(err, os.ErrNotExist) {
		return &state{}, nil
	} else if err != nil {
		return nil, err
	}
	var s state
	return &s, json.Unmarshal(data, &s)
}

func (s *state) save() error {
	if err := os.MkdirAll(stateDir(), 0o700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(stateDir(), "state.json"), append(data, '\n'), 0o600)
}

// backupContainer stores the full inspect output of a container.
func backupContainer(ctx context.Context, name string) (string, error) {
	out, err := docker(ctx, "inspect", "--type", "container", name)
	if err != nil {
		return "", err
	}
	dir := filepath.Join(stateDir(), "backups")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	path := filepath.Join(dir, fmt.Sprintf("%s-%s.json", name, time.Now().Format("20060102-150405")))
	return path, os.WriteFile(path, []byte(out+"\n"), 0o600)
}

func loadBackup(path string) (*containerInfo, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var infos []containerInfo
	if err := json.Unmarshal(data, &infos); err != nil {
		return nil, err
	}
	if len(infos) == 0 {
		return nil, fmt.Errorf("%s: empty backup", path)
	}
	return &infos[0], nil
}

// recreate replaces a container with a new spec. When the new container
// does not start, the previous one is created again from prev, with the
// image it ran (by ID: its tag may point to a new build by now).
func recreate(ctx context.Context, spec, prev containerSpec, prevImageID string) error {
	if prevImageID != "" {
		prev.Image = prevImageID
	}
	if _, err := docker(ctx, "rm", "-f", spec.Name); err != nil {
		return err
	}
	if _, err := docker(ctx, spec.runArgs()...); err != nil {
		_, _ = docker(ctx, "rm", "-f", spec.Name)
		if _, rerr := docker(ctx, prev.runArgs()...); rerr != nil {
			return fmt.Errorf("%w; restoring the previous container also failed: %v", err, rerr)
		}
		return fmt.Errorf("%w (the previous container was restored)", err)
	}
	return nil
}
