package cluster

import (
	"context"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

// GPUStats is the load of the GPUs of one machine, read with nvidia-smi.
// With several GPUs the utilisation is the average and power the sum.
type GPUStats struct {
	Util        float64 `json:"util"`
	Temperature float64 `json:"temperature,omitempty"`
	Power       float64 `json:"power,omitempty"`
}

// SystemMemory returns total and available memory from /proc/meminfo. On
// unified-memory machines (DGX Spark) this is also the GPU memory.
func SystemMemory() (total, available uint64, ok bool) {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, 0, false
	}
	for _, line := range strings.Split(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		v, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil {
			continue
		}
		switch fields[0] {
		case "MemTotal:":
			total = v << 10
		case "MemAvailable:":
			available = v << 10
		}
	}
	return total, available, total > 0
}

var (
	gpuMu   sync.Mutex
	gpuAt   time.Time
	gpuLast *GPUStats
)

// ReadGPU returns the GPU load, at most one nvidia-smi call every 3 seconds.
// It returns nil when nvidia-smi is not available.
func ReadGPU(ctx context.Context) *GPUStats {
	gpuMu.Lock()
	defer gpuMu.Unlock()
	if time.Since(gpuAt) < 3*time.Second {
		return gpuLast
	}
	gpuAt = time.Now()

	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits").Output()
	if err != nil {
		gpuLast = nil
		return nil
	}
	gpuLast = parseGPU(string(out))
	return gpuLast
}

func parseGPU(out string) *GPUStats {
	var s GPUStats
	n := 0
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		f := strings.Split(line, ",")
		if len(f) < 3 {
			continue
		}
		util, err := strconv.ParseFloat(strings.TrimSpace(f[0]), 64)
		if err != nil {
			continue
		}
		n++
		s.Util += util
		if t, err := strconv.ParseFloat(strings.TrimSpace(f[1]), 64); err == nil {
			s.Temperature = max(s.Temperature, t)
		}
		if p, err := strconv.ParseFloat(strings.TrimSpace(f[2]), 64); err == nil {
			s.Power += p
		}
	}
	if n == 0 {
		return nil
	}
	s.Util /= float64(n)
	return &s
}
