package server

import (
	"context"
	"net/http"
	"regexp"
	"strconv"
	"sync"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/cluster"
	"github.com/ollama/ollama/types/model"
)

// ClusterHistoryHandler returns the load and the requests of the last
// ?minutes= (default 60, at most one day).
func (s *Server) ClusterHistoryHandler(c *gin.Context) {
	minutes, err := strconv.Atoi(c.DefaultQuery("minutes", "60"))
	if err != nil || minutes < 1 {
		minutes = 60
	}
	minutes = min(minutes, 24*60)
	c.JSON(http.StatusOK, cluster.Recorder.History(time.Duration(minutes)*time.Minute))
}

var samplerOnce sync.Once

// startClusterSampler records the load of the cluster every few seconds
// for the history charts of the /cluster page.
func (s *Server) startClusterSampler() {
	samplerOnce.Do(func() {
		go func() {
			t := time.NewTicker(cluster.SampleInterval)
			defer t.Stop()
			for range t.C {
				cluster.Recorder.AddSample(s.clusterSample(context.Background()))
			}
		}()
	})
}

func (s *Server) clusterSample(ctx context.Context) cluster.Sample {
	smp := cluster.Sample{T: time.Now()}

	here := cluster.MachineSample{Name: "this machine", Util: -1}
	if total, avail, ok := cluster.SystemMemory(); ok {
		here.Total, here.Used = total, total-avail
	}
	if g := cluster.ReadGPU(ctx); g != nil {
		here.Util, here.Temp, here.Power = g.Util, g.Temperature, g.Power
	}
	smp.Machines = append(smp.Machines, here)

	if cfg, err := cluster.LoadConfig(); err == nil && len(cfg.ActiveWorkers()) > 0 {
		ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
		for _, c := range cluster.CheckWorkers(ctx, cfg) {
			m := cluster.MachineSample{Name: c.Worker.Label(), Util: -1}
			if c.Status != nil {
				for _, d := range c.Status.Devices {
					m.Total += d.TotalMemory
					m.Used += d.TotalMemory - min(d.FreeMemory, d.TotalMemory)
				}
				if g := c.Status.GPU; g != nil {
					m.Util, m.Temp, m.Power = g.Util, g.Temperature, g.Power
				}
			}
			smp.Machines = append(smp.Machines, m)
		}
		cancel()
	}

	smp.InFlight, smp.Active = cluster.Recorder.InFlightByModel()
	if len(smp.Active) == 0 {
		smp.Active = nil
	}

	if s.sched != nil {
		s.sched.loadedMu.Lock()
		refs := make([]*runnerRef, 0, len(s.sched.loaded))
		for _, r := range s.sched.loaded {
			refs = append(refs, r)
		}
		s.sched.loadedMu.Unlock()
		for _, r := range refs {
			// a loading model holds refMu until it runs, which can take
			// minutes; the sample goes on without it
			if !r.refMu.TryLock() {
				continue
			}
			name := r.modelKey
			if r.model != nil {
				name = model.ParseName(r.model.ShortName).DisplayShortest()
			}
			size := r.totalSize
			r.refMu.Unlock()
			if smp.Loaded == nil {
				smp.Loaded = map[string]uint64{}
			}
			smp.Loaded[name] = size
		}
	}
	return smp
}

// tailWriter keeps the end of a response, where generation endpoints put
// their token counts (the last chunk of a stream, or the end of one JSON).
type tailWriter struct {
	gin.ResponseWriter
	tail []byte

	// writes, first and last time data was written: for a stream the time
	// between the first and the last chunk is the generation time
	writes      int
	first, last time.Time
}

// tokens returns the token counts of the response. When the response does
// not report the generation time (OpenAI-compatible endpoints) and was
// streamed, the time from the first to the last chunk stands in for it.
func (w *tailWriter) tokens() cluster.Tokens {
	t := tokensFrom(w.tail)
	if t.GenTokens > 0 && t.GenDuration == 0 && w.writes > 2 {
		t.GenDuration = w.last.Sub(w.first)
	}
	return t
}

const tailSize = 4 << 10

func (w *tailWriter) Write(b []byte) (int, error) {
	w.keep(b)
	return w.ResponseWriter.Write(b)
}

func (w *tailWriter) WriteString(s string) (int, error) {
	w.keep([]byte(s))
	return w.ResponseWriter.WriteString(s)
}

func (w *tailWriter) keep(b []byte) {
	now := time.Now()
	if w.writes == 0 {
		w.first = now
	}
	w.writes++
	w.last = now
	if len(b) >= tailSize {
		w.tail = append(w.tail[:0], b[len(b)-tailSize:]...)
		return
	}
	w.tail = append(w.tail, b...)
	if len(w.tail) > 2*tailSize {
		w.tail = append(w.tail[:0], w.tail[len(w.tail)-tailSize:]...)
	}
}

var (
	// Ollama API: /api/generate, /api/chat
	evalCount       = regexp.MustCompile(`"eval_count":\s*(\d+)`)
	evalDuration    = regexp.MustCompile(`"eval_duration":\s*(\d+)`)
	promptEvalCount = regexp.MustCompile(`"prompt_eval_count":\s*(\d+)`)
	// OpenAI-compatible API: /v1/chat/completions, /v1/completions, /v1/responses
	completionTokens = regexp.MustCompile(`"(?:completion_tokens|output_tokens)":\s*(\d+)`)
	promptTokens     = regexp.MustCompile(`"(?:prompt_tokens|input_tokens)":\s*(\d+)`)
)

// tokensFrom reads the token counts from the end of a response.
func tokensFrom(tail []byte) cluster.Tokens {
	last := func(re *regexp.Regexp) int {
		all := re.FindAllSubmatch(tail, -1)
		if len(all) == 0 {
			return 0
		}
		n, _ := strconv.Atoi(string(all[len(all)-1][1]))
		return n
	}
	var t cluster.Tokens
	if n := last(evalCount); n > 0 {
		t.GenTokens = n
		t.GenDuration = time.Duration(last(evalDuration))
		t.PromptTokens = last(promptEvalCount)
		return t
	}
	t.GenTokens = last(completionTokens)
	t.PromptTokens = last(promptTokens)
	return t
}
