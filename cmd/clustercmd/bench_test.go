package clustercmd

import (
	"bytes"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestMeanStd(t *testing.T) {
	mean, std := meanStd([]float64{10, 12, 14})
	if mean != 12 || std != 2 {
		t.Fatalf("meanStd = %v ± %v, want 12 ± 2", mean, std)
	}
	if formatRate([]float64{816.58}) != "816.58" || formatRate(nil) != "-" {
		t.Fatal("formatRate of one sample or none")
	}
	if got := formatRate([]float64{10, 12, 14}); got != "12.00 ± 2.00" {
		t.Fatalf("formatRate = %q", got)
	}
}

func TestPPPromptIsNewAndAdjusts(t *testing.T) {
	b := &bencher{ppTokens: 512}
	first, second := b.ppPrompt(), b.ppPrompt()
	if first[:8] == second[:8] {
		t.Fatal("each pp prompt must start differently, or the runner reuses the cached prefix")
	}
	if n := strings.Count(first, " the"); n != 504 {
		t.Fatalf("first prompt has %d words, want 504", n)
	}
	// the model counted 509 tokens: three more words next time
	b.reps += b.ppTokens - 509
	if n := strings.Count(b.ppPrompt(), " the"); n != 507 {
		t.Fatalf("adjusted prompt has %d words, want 507", n)
	}
}

func TestPrintBench(t *testing.T) {
	runs := []benchRun{
		{label: "this machine, cold", cold: true, load: 3 * time.Second, pp: []float64{800}, tg: []float64{12}, ppTokens: 512, tgTokens: 128, local: 1 << 30, context: 4096},
		{label: "this machine, warm", pp: []float64{810, 820, 830}, tg: []float64{12, 12.5, 13}, ppTokens: 510, tgTokens: 90, local: 1 << 30, context: 4096},
		{label: "two machines, cold", err: errors.New("worker busy\nmore")},
	}
	var out bytes.Buffer
	printBench(&out, 512, 128, 3, runs)
	got := out.String()
	for _, want := range []string{"pp512 t/s", "tg128 t/s", "820.00 ± 10.00", "12.50 ± 0.50", "worker busy", "the prompt had 510 tokens", "stopped after 90 tokens"} {
		if !strings.Contains(got, want) {
			t.Errorf("output lacks %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "more") {
		t.Errorf("only the first line of an error is printed:\n%s", got)
	}

	res := benchJSON("m", "hw", "b1", 512, 128, runs)
	if r := res.Runs[1]; r.Samples != 3 || r.PromptTPS != 820 || r.GenStd != 0.5 || r.PromptTok != 510 {
		t.Fatalf("JSON run %+v", r)
	}
}
