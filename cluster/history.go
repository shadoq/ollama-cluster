package cluster

import (
	"sort"
	"time"
)

// SampleInterval is how often the server records the load of the cluster.
const SampleInterval = 5 * time.Second

// Sample is the load of the cluster at one moment.
type Sample struct {
	T        time.Time       `json:"t"`
	Machines []MachineSample `json:"machines"`
	// InFlight is the number of API requests running; Active splits the
	// ones with a model by model.
	InFlight int            `json:"in_flight"`
	Active   map[string]int `json:"active,omitempty"`
	// Loaded is the memory each loaded model takes.
	Loaded map[string]uint64 `json:"loaded,omitempty"`
}

// MachineSample is the memory and GPU load of one machine. Util is -1 when
// the GPU load is unknown.
type MachineSample struct {
	Name  string  `json:"name"`
	Used  uint64  `json:"used"`
	Total uint64  `json:"total"`
	Util  float64 `json:"util"`
	Temp  float64 `json:"temp,omitempty"`
	Power float64 `json:"power,omitempty"`
}

// AddSample records a sample.
func (a *Activity) AddSample(s Sample) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.samples = append(a.samples, s)
	a.samples = trimOld(a.samples, int(keepHistory/SampleInterval)+10, func(s Sample) time.Time { return s.T })
}

// InFlightByModel returns the number of requests running and, of those with
// a model, how many per model.
func (a *Activity) InFlightByModel() (int, map[string]int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	by := map[string]int{}
	for _, r := range a.inFlight {
		if r.Model != "" {
			by[r.Model]++
		}
	}
	return len(a.inFlight), by
}

// HistoryResponse is returned by GET /api/cluster/history.
type HistoryResponse struct {
	From time.Time `json:"from"`
	To   time.Time `json:"to"`

	// Samples are averaged over SampleStep seconds (in-flight requests take
	// the maximum, loaded models the last value).
	SampleStep int      `json:"sample_step"`
	Samples    []Sample `json:"samples"`

	// Buckets count the finished requests per BucketStep seconds and model.
	BucketStep int             `json:"bucket_step"`
	Buckets    []RequestBucket `json:"buckets"`

	// Models are all models in the range, the busiest first.
	Models []string `json:"models"`
}

// RequestBucket holds the requests finished in one time step.
type RequestBucket struct {
	T      time.Time              `json:"t"`
	Models map[string]*ModelStats `json:"models"`
}

// ModelStats sums up the requests of one model in a bucket. TokPerSec is the
// generation speed of the requests that report their generation time (Ollama
// API, and streamed OpenAI-compatible responses).
type ModelStats struct {
	Requests     int     `json:"requests"`
	Errors       int     `json:"errors,omitempty"`
	AvgSeconds   float64 `json:"avg_seconds"`
	PromptTokens int     `json:"prompt_tokens,omitempty"`
	GenTokens    int     `json:"gen_tokens,omitempty"`
	TokPerSec    float64 `json:"tok_per_sec,omitempty"`

	seconds   float64
	genTokens int
	genSec    float64
}

const (
	maxPoints  = 360
	maxBuckets = 60
)

// History returns the load and the requests of the last span.
func (a *Activity) History(span time.Duration) HistoryResponse {
	now := time.Now()
	from := now.Add(-span)

	a.mu.Lock()
	var samples []Sample
	for i := len(a.samples) - 1; i >= 0 && !a.samples[i].T.Before(from); i-- {
		samples = append(samples, a.samples[i])
	}
	// a request belongs to the range when it finished in it, also when it
	// started earlier (a long pull or generation)
	var done []Request
	for i := len(a.done) - 1; i >= 0 && !a.done[i].End().Before(from); i-- {
		done = append(done, a.done[i])
	}
	a.mu.Unlock()

	// newest first above, oldest first below
	for i, j := 0, len(samples)-1; i < j; i, j = i+1, j-1 {
		samples[i], samples[j] = samples[j], samples[i]
	}

	resp := HistoryResponse{From: from, To: now, Samples: []Sample{}, Buckets: []RequestBucket{}, Models: []string{}}

	step := max(SampleInterval, span/maxPoints).Truncate(time.Second)
	resp.SampleStep = int(step / time.Second)
	resp.Samples = downsample(samples, from, step)

	bstep := max(15*time.Second, span/maxBuckets).Truncate(time.Second)
	resp.BucketStep = int(bstep / time.Second)
	count := map[string]int{}
	n := int(span/bstep) + 1
	for i := 0; i < n; i++ {
		resp.Buckets = append(resp.Buckets, RequestBucket{T: from.Add(time.Duration(i) * bstep), Models: map[string]*ModelStats{}})
	}
	for _, r := range done {
		if r.Model == "" {
			continue
		}
		i := int(r.End().Sub(from) / bstep)
		if i < 0 || i >= n {
			continue
		}
		m := resp.Buckets[i].Models[r.Model]
		if m == nil {
			m = &ModelStats{}
			resp.Buckets[i].Models[r.Model] = m
		}
		m.Requests++
		if r.Status >= 500 {
			m.Errors++
		}
		m.seconds += r.Duration.Seconds()
		m.PromptTokens += r.PromptTokens
		m.GenTokens += r.GenTokens
		// speed only from requests that tell their generation time; the
		// whole request time would include the prompt and the queue
		if r.GenTokens > 0 && r.GenDuration > 0 {
			m.genTokens += r.GenTokens
			m.genSec += r.GenDuration.Seconds()
		}
		count[r.Model]++
	}
	for _, b := range resp.Buckets {
		for _, m := range b.Models {
			m.AvgSeconds = m.seconds / float64(m.Requests)
			if m.genSec > 0 {
				m.TokPerSec = float64(m.genTokens) / m.genSec
			}
		}
	}

	for _, s := range resp.Samples {
		for m := range s.Active {
			count[m] += 0
		}
		for m := range s.Loaded {
			count[m] += 0
		}
	}
	for m := range count {
		resp.Models = append(resp.Models, m)
	}
	sort.Slice(resp.Models, func(i, j int) bool {
		if count[resp.Models[i]] != count[resp.Models[j]] {
			return count[resp.Models[i]] > count[resp.Models[j]]
		}
		return resp.Models[i] < resp.Models[j]
	})
	return resp
}

// downsample merges samples into steps: memory, GPU load, temperature and
// power are averaged, requests running take the maximum and loaded models
// the last value. Steps without samples are left out.
func downsample(samples []Sample, from time.Time, step time.Duration) []Sample {
	var out []Sample
	var cur []Sample
	flush := func() {
		if len(cur) == 0 {
			return
		}
		merged := Sample{T: cur[len(cur)-1].T, Active: map[string]int{}, Loaded: cur[len(cur)-1].Loaded}
		type acc struct {
			m           MachineSample
			used, util  float64
			temp, power float64
			n, utilN    int
		}
		var order []string
		accs := map[string]*acc{}
		for _, s := range cur {
			merged.InFlight = max(merged.InFlight, s.InFlight)
			for k, v := range s.Active {
				merged.Active[k] = max(merged.Active[k], v)
			}
			for _, m := range s.Machines {
				a := accs[m.Name]
				if a == nil {
					a = &acc{m: m}
					accs[m.Name] = a
					order = append(order, m.Name)
				}
				a.n++
				a.used += float64(m.Used)
				a.m.Total = m.Total
				if m.Util >= 0 {
					a.utilN++
					a.util += m.Util
					a.temp += m.Temp
					a.power += m.Power
				}
			}
		}
		for _, name := range order {
			a := accs[name]
			m := MachineSample{Name: name, Total: a.m.Total, Used: uint64(a.used / float64(a.n)), Util: -1}
			if a.utilN > 0 {
				m.Util = a.util / float64(a.utilN)
				m.Temp = a.temp / float64(a.utilN)
				m.Power = a.power / float64(a.utilN)
			}
			merged.Machines = append(merged.Machines, m)
		}
		if len(merged.Active) == 0 {
			merged.Active = nil
		}
		out = append(out, merged)
		cur = cur[:0]
	}
	bucket := -1
	for _, s := range samples {
		i := int(s.T.Sub(from) / step)
		if i != bucket {
			flush()
			bucket = i
		}
		cur = append(cur, s)
	}
	flush()
	if out == nil {
		out = []Sample{}
	}
	return out
}
