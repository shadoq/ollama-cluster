package server

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/cluster"
	"github.com/ollama/ollama/ml"
)

func TestClusterActivityMiddleware(t *testing.T) {
	gin.SetMode(gin.TestMode)
	cluster.Recorder = cluster.NewActivity()

	var got string
	r := gin.New()
	r.Use(clusterActivityMiddleware())
	r.POST("/api/chat", func(c *gin.Context) {
		b, _ := io.ReadAll(c.Request.Body)
		got = string(b)
		c.String(http.StatusOK, `{"message":{"content":"hi"},"done":true,"prompt_eval_count":12,"eval_count":40,"eval_duration":2000000000}`)
	})
	r.GET("/api/cluster/activity", (&Server{}).ClusterActivityHandler)
	r.GET("/cluster", (&Server{}).ClusterPageHandler)

	// the handler still gets the whole body, also past the peeked prefix
	body := `{"model":"qwen3.5:9b","messages":[{"role":"user","content":"` + strings.Repeat("x", peekLimit) + `"}]}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body)))
	if w.Code != http.StatusOK || got != body {
		t.Fatalf("body changed: code %d, len %d want %d", w.Code, len(got), len(body))
	}

	// the page and the polling are not recorded
	for _, p := range []string{"/cluster", "/api/cluster/activity"} {
		w = httptest.NewRecorder()
		r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, p, nil))
		if w.Code != http.StatusOK {
			t.Fatalf("%s: %d", p, w.Code)
		}
	}
	if !strings.Contains(w.Body.String(), `"qwen3.5:9b"`) {
		t.Fatalf("activity: %s", w.Body.String())
	}

	snap := cluster.Recorder.Snapshot()
	if snap.Total != 1 || len(snap.Recent) != 1 || len(snap.InFlight) != 0 {
		t.Fatalf("snapshot: %+v", snap)
	}
	r0 := snap.Recent[0]
	if r0.Model != "qwen3.5:9b" || r0.Path != "/api/chat" || r0.Status != http.StatusOK || r0.GenTokens != 40 || r0.PromptTokens != 12 || r0.GenDuration != 2*time.Second {
		t.Fatalf("request: %+v", r0)
	}
	if snap.ByModel["qwen3.5:9b"] != 1 {
		t.Fatalf("by model: %+v", snap.ByModel)
	}
}

func TestClusterPage(t *testing.T) {
	if !strings.Contains(string(clusterPage), "api/cluster/activity") || !strings.Contains(string(clusterPage), "api/cluster/status") || !strings.Contains(string(clusterPage), "api/cluster/history") {
		t.Fatal("the page does not poll the cluster endpoints")
	}
}

func TestTokensFrom(t *testing.T) {
	stream := []byte(`{"response":"a","done":false}` + "\n" + `{"response":"","done":true,"prompt_eval_count":7,"eval_count":3,"eval_duration":1500}` + "\n")
	if got := tokensFrom(stream); got.GenTokens != 3 || got.PromptTokens != 7 || got.GenDuration != 1500 {
		t.Fatalf("ollama: %+v", got)
	}
	openai := []byte(`data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":21,"total_tokens":30}}` + "\n\ndata: [DONE]\n")
	if got := tokensFrom(openai); got.GenTokens != 21 || got.PromptTokens != 9 || got.GenDuration != 0 {
		t.Fatalf("openai: %+v", got)
	}
	if got := tokensFrom([]byte(`{"models":[]}`)); got != (cluster.Tokens{}) {
		t.Fatalf("none: %+v", got)
	}
}

func TestTailWriterKeepsTheEnd(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	tw := &tailWriter{ResponseWriter: c.Writer}
	for i := 0; i < 1000; i++ {
		_, _ = tw.Write([]byte(`{"response":"xxxxxxxxxx","done":false}` + "\n"))
	}
	_, _ = tw.WriteString(`{"done":true,"eval_count":5,"eval_duration":10}`)
	if len(tw.tail) > 2*tailSize || tokensFrom(tw.tail).GenTokens != 5 {
		t.Fatalf("tail %d bytes, tokens %+v", len(tw.tail), tokensFrom(tw.tail))
	}
	if w.Body.Len() < 1000*30 {
		t.Fatal("the response was not passed through")
	}
}

func TestTailWriterStreamTime(t *testing.T) {
	gin.SetMode(gin.TestMode)
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	tw := &tailWriter{ResponseWriter: c.Writer}
	for _, chunk := range []string{`data: {"choices":[{"delta":{"content":"a"}}]}`, `data: {"choices":[{"delta":{"content":"b"}}]}`, `data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}`} {
		_, _ = tw.WriteString(chunk)
		time.Sleep(5 * time.Millisecond)
	}
	if got := tw.tokens(); got.GenTokens != 2 || got.GenDuration < 5*time.Millisecond {
		t.Fatalf("stream: %+v", got)
	}

	// one write: no generation time
	c, _ = gin.CreateTestContext(httptest.NewRecorder())
	tw = &tailWriter{ResponseWriter: c.Writer}
	_, _ = tw.WriteString(`{"usage":{"prompt_tokens":3,"completion_tokens":2}}`)
	if got := tw.tokens(); got.GenTokens != 2 || got.GenDuration != 0 {
		t.Fatalf("single: %+v", got)
	}
}

func TestClusterLocalOnly(t *testing.T) {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.PUT("/x", clusterLocalOnly, func(c *gin.Context) { c.Status(http.StatusOK) })
	try := func(addr, host, origin string) int {
		req := httptest.NewRequest(http.MethodPut, "/x", nil)
		req.RemoteAddr, req.Host = addr, host
		req.Header.Set("X-Forwarded-For", "127.0.0.1")
		if origin != "" {
			req.Header.Set("Origin", origin)
		}
		w := httptest.NewRecorder()
		r.ServeHTTP(w, req)
		return w.Code
	}
	for _, c := range []struct {
		addr, host, origin string
		want               int
	}{
		{"127.0.0.1:5000", "127.0.0.1:11434", "", 200},
		{"[::1]:5000", "localhost:11434", "", 200},
		{"192.168.5.99:5000", "192.168.5.60:11434", "", 403},
		{"172.17.0.1:5000", "127.0.0.1:11434", "", 403},
		{"127.0.0.1:5000", "evil.example:11434", "", 403},
		{"127.0.0.1:5000", "127.0.0.1:11434", "http://evil.example", 403},
	} {
		if got := try(c.addr, c.host, c.origin); got != c.want {
			t.Errorf("%+v: %d", c, got)
		}
	}
	t.Setenv("OLLAMA_CLUSTER_ADMIN_ADDRS", "172.17.0.1")
	if got := try("172.17.0.1:5000", "127.0.0.1:11434", ""); got != 200 {
		t.Errorf("allowed address: %d", got)
	}
}

func TestActivityModelFromLongBody(t *testing.T) {
	gin.SetMode(gin.TestMode)
	cluster.Recorder = cluster.NewActivity()
	r := gin.New()
	r.Use(clusterActivityMiddleware())
	var got int
	r.POST("/v1/chat/completions", func(c *gin.Context) {
		b, _ := io.ReadAll(c.Request.Body)
		got = len(b)
		c.Status(http.StatusOK)
	})
	r.POST("/boom", func(c *gin.Context) { panic(http.ErrAbortHandler) })

	// OpenAI SDKs put the model after the messages; tool names are not models
	body := `{"messages":[{"role":"user","content":"` + strings.Repeat("x", 100<<10) + `"}],"tools":[{"function":{"name":"get_weather"}}],"model":"qwen3.8:27b"}`
	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
	if got != len(body) {
		t.Fatalf("handler read %d of %d bytes", got, len(body))
	}
	func() {
		defer func() { _ = recover() }()
		r.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, "/boom", strings.NewReader(`{}`)))
	}()

	s := cluster.Recorder.Snapshot()
	if len(s.InFlight) != 0 || len(s.Recent) != 2 {
		t.Fatalf("in flight %d, recent %d", len(s.InFlight), len(s.Recent))
	}
	if s.Recent[1].Model != "qwen3.8:27b" || s.Recent[0].Status != http.StatusInternalServerError {
		t.Fatalf("recent: %+v", s.Recent)
	}
}

type fakeVRAM map[ml.DeviceID]uint64

func (f fakeVRAM) VRAMByGPU(id ml.DeviceID) uint64 { return f[id] }

func TestRunnerDevicesKeepsBusyWorker(t *testing.T) {
	local := ml.DeviceID{ID: "0", Library: "CUDA"}
	worker := ml.DeviceID{ID: "10.100.48.2:50052/0", Library: cluster.RemoteLibrary}
	vram := fakeVRAM{local: 1 << 30, worker: 26 << 30}

	// the worker holds the model, so the current device list leaves it out
	current := []ml.DeviceInfo{{DeviceID: local, Name: "GB10"}}
	got := runnerDevices(vram, []ml.DeviceID{local, worker}, current)
	if len(got) != 2 || got[0].Name != "GB10" || got[0].Remote || !got[1].Remote || got[1].Size != 26<<30 || got[1].Name != worker.ID {
		t.Fatalf("runnerDevices = %+v", got)
	}

	// a runner without recorded devices falls back to the current list
	got = runnerDevices(vram, nil, current)
	if len(got) != 1 || got[0].Remote {
		t.Fatalf("runnerDevices without recorded devices = %+v", got)
	}
}
