// Package cluster lets one Ollama server use the memory of a second machine.
//
// The second machine runs a worker: the ggml RPC server from the same
// llama.cpp build that Ollama ships (ollama cluster worker). Ollama sees each
// worker device as an extra GPU with library "RPC" and llama-server reaches it
// with --rpc. The scheduler only uses a worker when a model does not fit on the
// local GPUs, and a worker that is down, busy or built from a different
// version is skipped, so the local machine never depends on the second one.
//
// The setup lives in cluster.json next to the models (OLLAMA_MODELS), so a
// Docker bind mount of the models directory carries it into the container and
// the `ollama cluster` commands on the host edit the same file.
package cluster

import (
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/ollama/ollama/envconfig"
)

const (
	DefaultRPCPort    = 50052
	DefaultStatusPort = 50053

	// PolicyAuto places a model on remote workers only when it does not fit
	// on the local GPUs. PolicySpread always splits models across all devices.
	// PolicyWorkerFirst puts models from WorkerFirstFrom up on the worker: the
	// whole model when it fits there, otherwise the worker is filled first.
	PolicyAuto        = "auto"
	PolicySpread      = "spread"
	PolicyWorkerFirst = "worker-first"

	// Placements of one model, decided by the policy or a rule in Models.
	PlaceAuto   = "auto"   // this machine when it fits, else split
	PlaceHere   = "here"   // this machine only
	PlaceSplit  = "split"  // across this machine and the worker
	PlaceWorker = "worker" // on the worker; split with the worker filled first when it does not fit there
)

// Config is the persistent cluster setup.
type Config struct {
	Workers   []Worker   `json:"workers,omitempty"`
	Locations []Location `json:"locations,omitempty"`

	// Policy is PolicyAuto (default), PolicySpread or PolicyWorkerFirst.
	Policy string `json:"policy,omitempty"`

	// WorkerFirstFrom is the model size (memory it needs, in bytes) from
	// which PolicyWorkerFirst sends a model to the worker. 0 means every model.
	WorkerFirstFrom uint64 `json:"worker_first_from,omitempty"`

	// KeepFreeHere is memory on this machine, in bytes, that models never
	// take, e.g. for other software running here. A model that does not fit
	// in the rest is split, and a split model leaves it free.
	KeepFreeHere uint64 `json:"keep_free_here,omitempty"`

	// Models fixes the placement of single models: PlaceHere, PlaceSplit,
	// PlaceWorker or PlaceAuto, by model name.
	Models map[string]string `json:"models,omitempty"`
}

// Worker is a remote machine that lends its GPU to this Ollama.
type Worker struct {
	// Name is a short label shown in status output, e.g. the hostname.
	Name string `json:"name"`

	// SSH is the user@host used by `ollama cluster` to manage the worker.
	SSH string `json:"ssh,omitempty"`

	// Addr is the address of the worker on the direct link. The RPC server
	// has no authentication, so it only ever listens on this address.
	Addr string `json:"addr"`

	// LocalAddr is the address of this machine on the same link; the worker
	// firewall admits only this address.
	LocalAddr string `json:"local_addr,omitempty"`

	RPCPort    int `json:"rpc_port,omitempty"`
	StatusPort int `json:"status_port,omitempty"`

	// Mode is how the worker is installed: "docker" or "native".
	Mode string `json:"mode,omitempty"`

	// Image is the Docker image the worker runs (docker mode).
	Image string `json:"image,omitempty"`

	// Cache enables the RPC tensor cache of the worker (add --cache).
	Cache bool `json:"cache,omitempty"`

	// Disabled keeps the worker configured but unused.
	Disabled bool `json:"disabled,omitempty"`
}

// Location is a directory holding Hugging Face repository copies. Locations
// are ordered from the best (fastest) to the worst.
type Location struct {
	Path string `json:"path"`

	// Archive marks storage that is too slow to load models from (e.g. a NAS).
	// Copies there are kept, but Ollama never links to them.
	Archive bool `json:"archive,omitempty"`
}

// Endpoint is the host:port of the worker RPC server.
func (w Worker) Endpoint() string {
	return net.JoinHostPort(w.Addr, strconv.Itoa(w.rpcPort()))
}

// StatusAddr is the host:port of the worker status server.
func (w Worker) StatusAddr() string {
	return net.JoinHostPort(w.Addr, strconv.Itoa(w.statusPort()))
}

func (w Worker) rpcPort() int {
	if w.RPCPort > 0 {
		return w.RPCPort
	}
	return DefaultRPCPort
}

func (w Worker) statusPort() int {
	if w.StatusPort > 0 {
		return w.StatusPort
	}
	return DefaultStatusPort
}

// Label is the worker name, falling back to its address.
func (w Worker) Label() string {
	if w.Name != "" {
		return w.Name
	}
	return w.Addr
}

// ActiveWorkers returns the configured workers that are not disabled.
func (c *Config) ActiveWorkers() []Worker {
	var workers []Worker
	for _, w := range c.Workers {
		if !w.Disabled && w.Addr != "" {
			workers = append(workers, w)
		}
	}
	return workers
}

// ValidPlacement reports whether p is a placement of a model.
func ValidPlacement(p string) bool {
	switch p {
	case PlaceAuto, PlaceHere, PlaceSplit, PlaceWorker:
		return true
	}
	return false
}

// Validate checks the workers, the locations, the policy and the model rules.
func (c *Config) Validate() error {
	for _, w := range c.Workers {
		if err := w.Validate(); err != nil {
			return err
		}
	}
	if err := validateLocations(c.Locations); err != nil {
		return err
	}
	switch c.Policy {
	case "", PolicyAuto, PolicySpread, PolicyWorkerFirst:
	default:
		return fmt.Errorf("policy must be %s, %s or %s", PolicyAuto, PolicySpread, PolicyWorkerFirst)
	}
	for m, p := range c.Models {
		if !ValidPlacement(p) {
			return fmt.Errorf("model %s: placement must be %s, %s, %s or %s", m, PlaceAuto, PlaceHere, PlaceSplit, PlaceWorker)
		}
	}
	return nil
}

// Rule returns the placement rule of a model, matching "name" and
// "name:latest" alike.
func (c *Config) Rule(model string) (string, bool) {
	if p, ok := c.Models[model]; ok {
		return p, true
	}
	if base, ok := strings.CutSuffix(model, ":latest"); ok {
		p, ok := c.Models[base]
		return p, ok
	}
	p, ok := c.Models[model+":latest"]
	return p, ok
}

// Placement decides where a model of the given size goes: a rule for the
// model wins, then the policy.
func (c *Config) Placement(model string, size uint64) string {
	if p, ok := c.Rule(model); ok && p != PlaceAuto {
		return p
	}
	switch {
	case c.Policy == PolicyWorkerFirst && size >= c.WorkerFirstFrom:
		return PlaceWorker
	case c.Spread():
		return PlaceSplit
	}
	return PlaceAuto
}

// Spread reports whether every model should be split across all devices.
func (c *Config) Spread() bool {
	return c.Policy == PolicySpread || envconfig.SchedSpread()
}

// ConfigPath is OLLAMA_CLUSTER_CONFIG or cluster.json in the models directory.
func ConfigPath() string {
	if p := os.Getenv("OLLAMA_CLUSTER_CONFIG"); p != "" {
		return p
	}
	return filepath.Join(envconfig.Models(), "cluster.json")
}

// LoadConfig reads the cluster configuration. A missing file is an empty
// configuration, not an error.
func LoadConfig() (*Config, error) {
	return LoadConfigFile(ConfigPath())
}

// LoadConfigFile reads the cluster configuration from path.
func LoadConfigFile(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return &Config{}, nil
	} else if err != nil {
		return nil, err
	}

	var c Config
	if err := json.Unmarshal(data, &c); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return &c, nil
}

// Save writes the configuration atomically to ConfigPath.
func (c *Config) Save() error {
	return c.SaveFile(ConfigPath())
}

// SaveFile writes the configuration atomically to path.
func (c *Config) SaveFile(path string) error {
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, append(data, '\n'), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// LoadLocations returns the load locations (non-archive) in order.
func (c *Config) LoadLocations() []Location {
	var locs []Location
	for _, l := range c.Locations {
		if !l.Archive {
			locs = append(locs, l)
		}
	}
	return locs
}
