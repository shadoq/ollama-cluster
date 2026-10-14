package server

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/cluster"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/version"
)

// registerClusterRoutes adds the /api/cluster endpoints used by the
// `ollama cluster` commands. They run inside the server process, so in Docker
// they see the container's paths and write files as the container user.
func (s *Server) registerClusterRoutes(r *gin.Engine) {
	r.GET("/api/cluster/status", s.ClusterStatusHandler)
	r.PUT("/api/cluster/config", clusterLocalOnly, s.ClusterConfigHandler)
	r.POST("/api/cluster/link", clusterLocalOnly, s.ClusterLinkHandler)
	r.GET("/api/cluster/links", s.ClusterLinksHandler)
	r.POST("/api/cluster/relink", clusterLocalOnly, s.ClusterRelinkHandler)
	r.GET("/api/cluster/models", s.ClusterModelsHandler)
	r.GET("/api/cluster/activity", s.ClusterActivityHandler)
	r.GET("/api/cluster/history", s.ClusterHistoryHandler)
	r.GET("/cluster", s.ClusterPageHandler)
	r.GET("/status", s.ClusterPageHandler)

	cluster.CaptureLog()
}

// workerChecks keeps the last worker check for a second: the status endpoint
// has no authentication and each check talks to every worker. Callers at
// the same time wait for one check.
var workerChecks struct {
	sync.Mutex
	at     time.Time
	key    string
	checks []cluster.WorkerCheck
}

func cachedWorkerChecks(ctx context.Context, cfg *cluster.Config) []cluster.WorkerCheck {
	key, _ := json.Marshal(cfg.Workers)
	workerChecks.Lock()
	defer workerChecks.Unlock()
	if string(key) == workerChecks.key && time.Since(workerChecks.at) < time.Second {
		return slices.Clone(workerChecks.checks)
	}
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	checks := cluster.CheckWorkers(ctx, cfg)
	workerChecks.at, workerChecks.key, workerChecks.checks = time.Now(), string(key), checks
	return slices.Clone(checks)
}

func (s *Server) ClusterStatusHandler(c *gin.Context) {
	cfg, err := cluster.LoadConfig()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	resp := cluster.StatusResponse{
		Version:    version.Version,
		LlamaCpp:   cluster.LlamaCppVersion,
		ConfigPath: cluster.ConfigPath(),
		Config:     *cfg,
		Devices:    []cluster.DeviceStatus{},
		Workers:    []cluster.WorkerCheck{},
		Loaded:     []cluster.LoadedModel{},
	}

	resp.Workers = append(resp.Workers, cachedWorkerChecks(c.Request.Context(), cfg)...)

	if s.sched == nil {
		c.JSON(http.StatusOK, resp)
		return
	}

	s.sched.loadedMu.Lock()
	runners := make([]ml.FilteredRunnerDiscovery, 0, len(s.sched.loaded))
	refs := make([]*runnerRef, 0, len(s.sched.loaded))
	for _, r := range s.sched.loaded {
		runners = append(runners, r)
		refs = append(refs, r)
	}
	s.sched.loadedMu.Unlock()

	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Second)
	defer cancel()
	gpus := s.sched.getGpuFn(ctx, runners)
	for _, g := range gpus {
		resp.Devices = append(resp.Devices, cluster.DeviceStatus{
			ID:          g.ID,
			Library:     g.Library,
			Name:        g.Name,
			Description: g.Description,
			TotalMemory: g.TotalMemory,
			FreeMemory:  g.FreeMemory,
			Integrated:  g.Integrated,
			Remote:      cluster.IsRemote(g),
		})
	}

	for _, r := range refs {
		r.refMu.Lock()
		lm := cluster.LoadedModel{
			Model:        r.modelKey,
			TrainContext: r.trainContext,
			Parallel:     r.numParallel,
			Active:       int(r.refCount),
			Loading:      r.loading,
			Size:         r.totalSize,
			VRAM:         r.vramSize,
			ExpiresAt:    r.expiresAt,
			KeepAlive:    r.sessionDuration,
			PID:          r.pid,
		}
		if r.model != nil {
			lm.Model = model.ParseName(r.model.ShortName).DisplayShortest()
		}
		if r.Options != nil {
			lm.Context = r.Options.NumCtx
		}
		if r.llama != nil {
			lm.Context = r.llama.ContextLength()
			lm.Devices = runnerDevices(r.llama, r.gpus, gpus)
		}
		r.refMu.Unlock()
		resp.Loaded = append(resp.Loaded, lm)
	}

	c.JSON(http.StatusOK, resp)
}

// runnerDevices returns the memory of a runner per device. It uses the
// devices recorded when the model loaded: a worker that holds the model is
// reported busy, or misses a check while it receives the weights, and is
// then left out of the current device list.
func runnerDevices(r interface{ VRAMByGPU(ml.DeviceID) uint64 }, loaded []ml.DeviceID, current []ml.DeviceInfo) []cluster.DeviceMemory {
	names := map[ml.DeviceID]string{}
	ids := slices.Clone(loaded)
	for _, g := range current {
		names[g.DeviceID] = g.Name
		if !slices.Contains(ids, g.DeviceID) {
			ids = append(ids, g.DeviceID)
		}
	}
	var devices []cluster.DeviceMemory
	for _, id := range ids {
		size := r.VRAMByGPU(id)
		if size == 0 {
			continue
		}
		name := names[id]
		if name == "" {
			name = id.ID
		}
		devices = append(devices, cluster.DeviceMemory{Name: name, Remote: id.Library == cluster.RemoteLibrary, Size: size})
	}
	return devices
}

func (s *Server) ClusterConfigHandler(c *gin.Context) {
	var cfg cluster.Config
	if err := c.ShouldBindJSON(&cfg); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if err := cfg.Validate(); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if err := cfg.Save(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, cfg)
}

func (s *Server) ClusterLinkHandler(c *gin.Context) {
	var req cluster.LinkRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	cfg, err := cluster.LoadConfig()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	created, err := cluster.LinkBlob(cfg, req.Digest, req.Path)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, cluster.LinkResponse{Digest: req.Digest, Path: req.Path, Created: created})
}

func (s *Server) ClusterLinksHandler(c *gin.Context) {
	links, err := cluster.Links()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if links == nil {
		links = []cluster.Link{}
	}
	c.JSON(http.StatusOK, cluster.LinksResponse{Links: links})
}

func (s *Server) ClusterRelinkHandler(c *gin.Context) {
	cfg, err := cluster.LoadConfig()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	changes, err := cluster.Relink(cfg)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if changes == nil {
		changes = []cluster.RelinkResult{}
	}
	c.JSON(http.StatusOK, cluster.RelinkResponse{Changes: changes})
}

// ClusterModelsHandler lists the blobs of every model and where their data
// lives: in the Ollama store or, through a link, in a repository copy.
func (s *Server) ClusterModelsHandler(c *gin.Context) {
	manifests, err := manifest.Manifests(true)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	resp := cluster.ModelsResponse{Models: []cluster.ModelFiles{}}
	for name, mf := range manifests {
		if name.EqualFold(model.ParseName(manifest.RunnerLlamaCPP + ":" + mf.Digest())) {
			continue
		}
		digests, err := manifest.ReferencedBlobDigestsForName(name)
		if err != nil {
			continue
		}
		m := cluster.ModelFiles{Model: name.DisplayShortest(), Blobs: []cluster.BlobFile{}}
		for _, d := range digests {
			m.Blobs = append(m.Blobs, blobFile(d))
		}
		resp.Models = append(resp.Models, m)
	}
	sort.Slice(resp.Models, func(i, j int) bool { return resp.Models[i].Model < resp.Models[j].Model })
	c.JSON(http.StatusOK, resp)
}

func blobFile(digest string) cluster.BlobFile {
	b := cluster.BlobFile{Digest: digest}
	p, err := manifest.BlobsPath(digest)
	if err != nil {
		return b
	}
	fi, err := os.Lstat(p)
	if err != nil {
		return b
	}
	if fi.Mode()&os.ModeSymlink != 0 {
		target, err := os.Readlink(p)
		if err != nil {
			return b
		}
		if !filepath.IsAbs(target) {
			target = filepath.Join(filepath.Dir(p), target)
		}
		b.Target = target
		if fi, err = os.Stat(p); err != nil {
			return b
		}
	}
	b.Size = fi.Size()
	b.OK = true
	return b
}

//go:embed cluster_page.html
var clusterPage []byte

// ClusterPageHandler serves a page that shows the cluster and what the server
// is doing, refreshed every few seconds from the JSON endpoints.
func (s *Server) ClusterPageHandler(c *gin.Context) {
	c.Header("Cache-Control", "no-store")
	c.Data(http.StatusOK, "text/html; charset=utf-8", clusterPage)
}

func (s *Server) ClusterActivityHandler(c *gin.Context) {
	c.JSON(http.StatusOK, cluster.Recorder.Snapshot())
}

// quietPaths are not recorded: the page and the polling of monitoring tools.
var quietPaths = map[string]bool{
	"/":                     true,
	"/cluster":              true,
	"/status":               true,
	"/api/cluster/status":   true,
	"/api/cluster/activity": true,
	"/api/cluster/history":  true,
	"/api/cluster/models":   true,
	"/api/cluster/links":    true,
	"/api/version":          true,
	"/api/ps":               true,
}

// modelField and nameField find the model name in a JSON request body.
// "name" is a fallback for the model management endpoints; in chat requests
// it is also the name of tools and messages.
var (
	modelField = regexp.MustCompile(`"model"\s*:\s*"([^"]{1,200})"`)
	nameField  = regexp.MustCompile(`"name"\s*:\s*"([^"]{1,200})"`)
)

func findModel(body []byte, nameToo bool) string {
	if m := modelField.FindSubmatch(body); m != nil {
		return string(m[1])
	}
	if nameToo {
		if m := nameField.FindSubmatch(body); m != nil {
			return string(m[1])
		}
	}
	return ""
}

// peekLimit is how much of a request body is read before the handler to find
// the model name. Clients that put "model" after long messages (the OpenAI
// SDKs do) are covered by the end of the body, see bodyTail.
const peekLimit = 64 << 10

// bodyTail passes the rest of a request body through and, when it ends,
// looks for the model name in its last bytes.
type bodyTail struct {
	r      io.Reader
	tail   []byte
	onDone func([]byte)
}

func (b *bodyTail) Read(p []byte) (int, error) {
	n, err := b.r.Read(p)
	b.tail = append(b.tail, p[:n]...)
	if len(b.tail) > 8<<10 {
		b.tail = append(b.tail[:0], b.tail[len(b.tail)-4<<10:]...)
	}
	if err == io.EOF && b.onDone != nil {
		b.onDone(b.tail)
		b.onDone = nil
	}
	return n, err
}

// clientAddr is the peer address of the connection. Forwarded-for headers
// are not used: anyone could set them.
func clientAddr(r *http.Request) string {
	if host, _, err := net.SplitHostPort(r.RemoteAddr); err == nil {
		return host
	}
	return r.RemoteAddr
}

// clusterActivityMiddleware records every API request in cluster.Recorder:
// path, model, client, status and duration (for streaming responses the
// whole generation).
func clusterActivityMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		path := c.Request.URL.Path
		if quietPaths[path] || c.Request.Method == http.MethodOptions || c.Request.Method == http.MethodHead {
			c.Next()
			return
		}
		id := cluster.Recorder.Begin(cluster.Request{
			Method: c.Request.Method,
			Path:   path,
			Client: clientAddr(c.Request),
		})
		// clients often send JSON without a content type, so any POST body
		// except uploads is checked
		if c.Request.Method == http.MethodPost && c.Request.Body != nil && !strings.HasPrefix(path, "/api/blobs/") {
			nameToo := !strings.Contains(path, "chat") && !strings.Contains(path, "generate") && !strings.Contains(path, "completions") &&
				!strings.Contains(path, "messages") && !strings.Contains(path, "responses")
			prefix, _ := io.ReadAll(io.LimitReader(c.Request.Body, peekLimit))
			model := findModel(prefix, nameToo)
			var rest io.Reader = c.Request.Body
			if model != "" {
				cluster.Recorder.SetModel(id, model)
			} else if len(prefix) == peekLimit {
				rest = &bodyTail{r: c.Request.Body, onDone: func(tail []byte) {
					if m := findModel(tail, nameToo); m != "" {
						cluster.Recorder.SetModel(id, m)
					}
				}}
			}
			c.Request.Body = struct {
				io.Reader
				io.Closer
			}{io.MultiReader(bytes.NewReader(prefix), rest), c.Request.Body}
		}
		tw := &tailWriter{ResponseWriter: c.Writer}
		c.Writer = tw
		// recorded also when the handler panics (e.g. an aborted proxy copy)
		done := false
		defer func() {
			if !done {
				cluster.Recorder.End(id, http.StatusInternalServerError, tw.Size(), cluster.Tokens{})
			}
		}()
		c.Next()
		done = true
		cluster.Recorder.End(id, tw.Status(), tw.Size(), tw.tokens())
	}
}
