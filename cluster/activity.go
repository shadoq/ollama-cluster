package cluster

import (
	"context"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"sync"
	"time"
)

// Activity keeps what the server did recently in memory: requests in flight,
// the last finished requests and the last log entries. The /cluster page and
// GET /api/cluster/activity show it. Nothing is written to disk.
type Activity struct {
	mu       sync.Mutex
	started  time.Time
	nextID   uint64
	inFlight map[uint64]*Request
	done     []Request // finished requests of the last day, oldest first
	samples  []Sample  // load every few seconds, oldest first
	log      []LogEntry
	total    int
	errors   int
	byModel  map[string]int
}

// Request is one HTTP request to the API.
type Request struct {
	ID       uint64        `json:"id"`
	Start    time.Time     `json:"start"`
	Method   string        `json:"method"`
	Path     string        `json:"path"`
	Model    string        `json:"model,omitempty"`
	Client   string        `json:"client,omitempty"`
	Status   int           `json:"status,omitempty"`
	Duration time.Duration `json:"duration"`
	Bytes    int           `json:"bytes,omitempty"`
	Tokens
}

// Tokens are the token counts a generation response reports. GenDuration is
// the generation time without the prompt; it is zero when the response does
// not report it (OpenAI-compatible endpoints).
type Tokens struct {
	PromptTokens int           `json:"prompt_tokens,omitempty"`
	GenTokens    int           `json:"gen_tokens,omitempty"`
	GenDuration  time.Duration `json:"gen_duration,omitempty"`
}

// LogEntry is one server log record at level INFO or above.
type LogEntry struct {
	Time    time.Time `json:"time"`
	Level   string    `json:"level"`
	Message string    `json:"message"`
	Attrs   string    `json:"attrs,omitempty"`
}

// ActivityResponse is returned by GET /api/cluster/activity.
type ActivityResponse struct {
	StartedAt time.Time      `json:"started_at"`
	Now       time.Time      `json:"now"`
	Total     int            `json:"total"`
	Errors    int            `json:"errors"`
	ByModel   map[string]int `json:"by_model"`
	InFlight  []Request      `json:"in_flight"`
	Recent    []Request      `json:"recent"`
	Log       []LogEntry     `json:"log"`
}

const (
	keepRequests = 200 // shown as the last requests
	keepLog      = 300
	keepDone     = 50000
	keepHistory  = 24 * time.Hour

	// Request fields come from clients; long values are cut and the count
	// per model keeps at most maxModels names, the rest go to otherModels.
	maxField    = 256
	maxModels   = 500
	otherModels = "(other)"
)

func clip(s string) string {
	if len(s) <= maxField {
		return s
	}
	return strings.ToValidUTF8(s[:maxField], "") + "..."
}

// End is when the request finished.
func (r Request) End() time.Time {
	return r.Start.Add(r.Duration)
}

// Recorder is the activity of this server process.
var Recorder = NewActivity()

func NewActivity() *Activity {
	return &Activity{
		started:  time.Now(),
		inFlight: map[uint64]*Request{},
		byModel:  map[string]int{},
	}
}

// Begin records a request that has started and returns its id.
func (a *Activity) Begin(r Request) uint64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.nextID++
	r.ID = a.nextID
	r.Path, r.Client, r.Model = clip(r.Path), clip(r.Client), clip(r.Model)
	if r.Start.IsZero() {
		r.Start = time.Now()
	}
	a.inFlight[r.ID] = &r
	return r.ID
}

// SetModel adds the model name to a request in flight.
func (a *Activity) SetModel(id uint64, model string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if r, ok := a.inFlight[id]; ok {
		r.Model = clip(model)
	}
}

// End records the result of a request.
func (a *Activity) End(id uint64, status, bytes int, t Tokens) {
	a.mu.Lock()
	defer a.mu.Unlock()
	r, ok := a.inFlight[id]
	if !ok {
		return
	}
	delete(a.inFlight, id)
	r.Status = status
	r.Bytes = max(bytes, 0)
	r.Duration = time.Since(r.Start)
	r.Tokens = t
	a.total++
	if status >= 500 {
		a.errors++
	}
	if r.Model != "" {
		if _, ok := a.byModel[r.Model]; ok || len(a.byModel) < maxModels {
			a.byModel[r.Model]++
		} else {
			a.byModel[otherModels]++
		}
	}
	// appended when finished, so the list is ordered by end time
	a.done = append(a.done, *r)
	a.done = trimOld(a.done, keepDone, Request.End)
}

// trimOld drops entries beyond limitN and older than keepHistory. Entries are
// oldest first. Trimming copies the slice, so it waits until there is a
// batch to drop; readers filter by time anyway.
func trimOld[T any](s []T, limitN int, at func(T) time.Time) []T {
	cut := len(s) - limitN
	if cut < 0 {
		cut = 0
	}
	limit := time.Now().Add(-keepHistory)
	for cut < len(s) && at(s[cut]).Before(limit) {
		cut++
	}
	if cut < 1024 && len(s) <= limitN+limitN/10 {
		return s
	}
	return append(s[:0:0], s[cut:]...)
}

// Log records a log entry.
func (a *Activity) Log(e LogEntry) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.log = append(a.log, e)
	if len(a.log) > keepLog {
		a.log = append([]LogEntry(nil), a.log[len(a.log)-keepLog:]...)
	}
}

// Snapshot returns the activity, newest first.
func (a *Activity) Snapshot() ActivityResponse {
	a.mu.Lock()
	defer a.mu.Unlock()
	now := time.Now()
	resp := ActivityResponse{
		StartedAt: a.started,
		Now:       now,
		Total:     a.total,
		Errors:    a.errors,
		ByModel:   map[string]int{},
		InFlight:  []Request{},
		Recent:    make([]Request, 0, keepRequests),
		Log:       make([]LogEntry, 0, len(a.log)),
	}
	for k, v := range a.byModel {
		resp.ByModel[k] = v
	}
	for _, r := range a.inFlight {
		c := *r
		c.Duration = now.Sub(c.Start)
		resp.InFlight = append(resp.InFlight, c)
	}
	sort.Slice(resp.InFlight, func(i, j int) bool { return resp.InFlight[i].Start.Before(resp.InFlight[j].Start) })
	for i := len(a.done) - 1; i >= 0 && len(resp.Recent) < keepRequests; i-- {
		resp.Recent = append(resp.Recent, a.done[i])
	}
	for i := len(a.log) - 1; i >= 0; i-- {
		resp.Log = append(resp.Log, a.log[i])
	}
	return resp
}

// --- log capture ------------------------------------------------------------

var teeOnce sync.Once

// CaptureLog makes the default logger also record INFO and above in
// Recorder. It is safe to call more than once.
func CaptureLog() {
	teeOnce.Do(func() {
		slog.SetDefault(slog.New(&teeHandler{inner: slog.Default().Handler(), rec: Recorder}))
	})
}

type teeHandler struct {
	inner slog.Handler
	rec   *Activity
	attrs []slog.Attr
	group string
}

func (h *teeHandler) Enabled(ctx context.Context, l slog.Level) bool {
	return l >= slog.LevelInfo || h.inner.Enabled(ctx, l)
}

func (h *teeHandler) Handle(ctx context.Context, r slog.Record) error {
	if r.Level >= slog.LevelInfo {
		var b strings.Builder
		write := func(a slog.Attr) {
			if b.Len() > 0 {
				b.WriteByte(' ')
			}
			fmt.Fprintf(&b, "%s=%v", a.Key, a.Value.Resolve())
		}
		for _, a := range h.attrs {
			write(a)
		}
		r.Attrs(func(a slog.Attr) bool {
			write(h.qualify(a))
			return true
		})
		h.rec.Log(LogEntry{Time: r.Time, Level: r.Level.String(), Message: r.Message, Attrs: b.String()})
	}
	if h.inner.Enabled(ctx, r.Level) {
		return h.inner.Handle(ctx, r)
	}
	return nil
}

// qualify prefixes the key with the current group, as text handlers do.
func (h *teeHandler) qualify(a slog.Attr) slog.Attr {
	if h.group != "" {
		a.Key = h.group + "." + a.Key
	}
	return a
}

func (h *teeHandler) WithAttrs(as []slog.Attr) slog.Handler {
	attrs := append([]slog.Attr(nil), h.attrs...)
	for _, a := range as {
		attrs = append(attrs, h.qualify(a))
	}
	return &teeHandler{inner: h.inner.WithAttrs(as), rec: h.rec, attrs: attrs, group: h.group}
}

func (h *teeHandler) WithGroup(name string) slog.Handler {
	g := name
	if h.group != "" {
		g = h.group + "." + name
	}
	return &teeHandler{inner: h.inner.WithGroup(name), rec: h.rec, attrs: h.attrs, group: g}
}
