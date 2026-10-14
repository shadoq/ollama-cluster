package cluster

import "time"

// Types of the /api/cluster endpoints, shared by the server and the
// `ollama cluster` commands.

// StatusResponse is returned by GET /api/cluster/status.
type StatusResponse struct {
	Version    string `json:"version"`
	LlamaCpp   string `json:"llama_cpp"`
	ConfigPath string `json:"config_path"`
	Config     Config `json:"config"`

	// Devices are all devices the scheduler sees, local and remote.
	Devices []DeviceStatus `json:"devices"`

	// Workers are the checks of the configured workers.
	Workers []WorkerCheck `json:"workers"`

	// Loaded are the models in memory and where they sit.
	Loaded []LoadedModel `json:"loaded"`
}

// DeviceStatus is a device as the scheduler sees it.
type DeviceStatus struct {
	ID          string `json:"id"`
	Library     string `json:"library"`
	Name        string `json:"name"`
	Description string `json:"description"`
	TotalMemory uint64 `json:"total_memory"`
	FreeMemory  uint64 `json:"free_memory"`
	Integrated  bool   `json:"integrated,omitempty"`
	Remote      bool   `json:"remote,omitempty"`
}

// LoadedModel is a loaded model and its memory per device.
type LoadedModel struct {
	Model   string         `json:"model"`
	Context int            `json:"context"`
	Devices []DeviceMemory `json:"devices"`

	// TrainContext is the context the model was trained with.
	TrainContext int `json:"train_context,omitempty"`
	// Parallel is the number of requests the runner serves at once.
	Parallel int `json:"parallel,omitempty"`
	// Active is the number of requests using the model right now.
	Active int `json:"active"`
	// Loading is set while the model is still being loaded.
	Loading bool `json:"loading,omitempty"`
	// Size is the memory the model takes in total, VRAM the part on GPUs.
	Size uint64 `json:"size,omitempty"`
	VRAM uint64 `json:"vram,omitempty"`
	// ExpiresAt is when an idle model is unloaded; KeepAlive below zero
	// means never.
	ExpiresAt time.Time     `json:"expires_at"`
	KeepAlive time.Duration `json:"keep_alive"`
	PID       int           `json:"pid,omitempty"`
}

// DeviceMemory is the memory a model holds on one device.
type DeviceMemory struct {
	Name   string `json:"name"`
	Remote bool   `json:"remote,omitempty"`
	Size   uint64 `json:"size"`
}
