package cluster

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"
	"time"
)

func TestActivityKeepsTheLast(t *testing.T) {
	a := NewActivity()
	for i := 0; i < keepRequests+10; i++ {
		id := a.Begin(Request{Method: "POST", Path: "/api/generate"})
		a.SetModel(id, "m")
		a.End(id, 200, 1, Tokens{GenTokens: 10, GenDuration: time.Second})
	}
	open := a.Begin(Request{Method: "POST", Path: "/api/chat", Model: "big"})
	s := a.Snapshot()
	if s.Total != keepRequests+10 || len(s.Recent) != keepRequests || s.ByModel["m"] != keepRequests+10 {
		t.Fatalf("total %d, recent %d, by model %v", s.Total, len(s.Recent), s.ByModel)
	}
	if len(s.InFlight) != 1 || s.InFlight[0].ID != open || s.InFlight[0].Model != "big" {
		t.Fatalf("in flight: %+v", s.InFlight)
	}
	if s.Recent[0].ID < s.Recent[1].ID {
		t.Fatal("recent requests are not newest first")
	}
	a.End(open, 500, 0, Tokens{})
	if a.Snapshot().Errors != 1 {
		t.Fatal("server error not counted")
	}
}

func TestTeeHandler(t *testing.T) {
	var out bytes.Buffer
	rec := NewActivity()
	l := slog.New(&teeHandler{inner: slog.NewTextHandler(&out, &slog.HandlerOptions{Level: slog.LevelWarn}), rec: rec})

	l.Debug("hidden")
	l.With("worker", "gx10").WithGroup("g").Info("cluster worker available", "endpoint", "10.100.48.2:50052")
	l.Warn("cluster worker skipped")

	log := rec.Snapshot().Log
	if len(log) != 2 || log[0].Level != "WARN" || log[1].Message != "cluster worker available" {
		t.Fatalf("log: %+v", log)
	}
	if log[1].Attrs != "worker=gx10 g.endpoint=10.100.48.2:50052" {
		t.Fatalf("attrs: %q", log[1].Attrs)
	}
	// the inner handler keeps its own level
	if strings.Contains(out.String(), "available") || !strings.Contains(out.String(), "skipped") {
		t.Fatalf("inner output: %s", out.String())
	}
}

func TestHistory(t *testing.T) {
	a := NewActivity()
	now := time.Now()
	for i := 0; i < 720; i++ { // one hour of samples
		ts := now.Add(-time.Hour + time.Duration(i)*SampleInterval)
		util := 20.0
		if i%2 == 1 {
			util = 40
		}
		a.samples = append(a.samples, Sample{
			T:        ts,
			Machines: []MachineSample{{Name: "this machine", Used: 10, Total: 100, Util: util}, {Name: "w", Used: 30, Total: 100, Util: -1}},
			InFlight: i % 3,
			Active:   map[string]int{"m": i % 3},
			Loaded:   map[string]uint64{"m": 5},
		})
	}
	for i := 0; i < 30; i++ {
		a.done = append(a.done, Request{Start: now.Add(-time.Duration(i) * time.Minute), Model: "m", Status: 200, Duration: 2 * time.Second, Tokens: Tokens{GenTokens: 100, GenDuration: time.Second}})
	}
	a.done = append(a.done, Request{Start: now.Add(-2 * time.Minute), Model: "other", Status: 500, Duration: time.Second, Tokens: Tokens{GenTokens: 50}})

	h := a.History(time.Hour)
	if h.SampleStep != 10 || len(h.Samples) < 355 || len(h.Samples) > 361 {
		t.Fatalf("step %d, %d samples", h.SampleStep, len(h.Samples))
	}
	s := h.Samples[len(h.Samples)/2]
	if s.Machines[0].Util != 30 || s.Machines[1].Util != -1 || s.Machines[0].Used != 10 || s.InFlight != 2 && s.InFlight != 1 {
		t.Fatalf("merged sample: %+v", s)
	}
	if h.BucketStep != 60 || len(h.Buckets) != 61 {
		t.Fatalf("bucket step %d, %d buckets", h.BucketStep, len(h.Buckets))
	}
	var reqs, errs int
	var speed float64
	for _, b := range h.Buckets {
		for name, m := range b.Models {
			reqs += m.Requests
			errs += m.Errors
			if name == "m" && m.TokPerSec > 0 {
				speed = m.TokPerSec
			}
			if name == "other" && (m.TokPerSec != 0 || m.GenTokens != 50) {
				t.Fatalf("speed without generation time: %+v", m)
			}
		}
	}
	if reqs != 31 || errs != 1 || speed != 100 {
		t.Fatalf("requests %d, errors %d, speed %v", reqs, errs, speed)
	}
	if len(h.Models) != 2 || h.Models[0] != "m" {
		t.Fatalf("models: %v", h.Models)
	}

	// a short range keeps only its own part
	if h := a.History(5 * time.Minute); len(h.Samples) == 0 || h.Samples[0].T.Before(now.Add(-5*time.Minute)) {
		t.Fatalf("short range: %d samples", len(h.Samples))
	}
}

func TestParseGPU(t *testing.T) {
	g := parseGPU("10, 50, 20.5\n30, 60, [N/A]\n")
	if g == nil || g.Util != 20 || g.Temperature != 60 || g.Power != 20.5 {
		t.Fatalf("%+v", g)
	}
	if parseGPU("") != nil {
		t.Fatal("empty output")
	}
}
