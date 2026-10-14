package server

import (
	"net"
	"net/http"
	"os"
	"strings"

	"github.com/gin-gonic/gin"
)

// clusterLocalOnly admits requests that change the cluster setup or model
// files only from the machine Ollama runs on: the config feeds commands that
// `ollama cluster` runs over SSH on the worker, and the rest of the Ollama
// API has no authentication. The peer address of the connection counts,
// never a forwarded-for header. Requests from a browser (with an Origin
// header) are refused, so a web page cannot send them through the user's
// browser, and so is a Host that is not an address or this machine's name
// (DNS rebinding). OLLAMA_CLUSTER_ADMIN_ADDRS adds addresses, e.g. the
// Docker bridge gateway when Ollama runs on a bridge network.
func clusterLocalOnly(c *gin.Context) {
	if !clusterAdminAllowed(c.Request) {
		c.AbortWithStatusJSON(http.StatusForbidden, gin.H{"error": "changes to the cluster setup are accepted only from the machine Ollama runs on (ollama cluster ... on that machine)"})
		return
	}
	c.Next()
}

func clusterAdminAllowed(r *http.Request) bool {
	if r.Header.Get("Origin") != "" {
		return false
	}
	host := r.Host
	if h, _, err := net.SplitHostPort(host); err == nil {
		host = h
	}
	host = strings.Trim(host, "[]")
	if name, _ := os.Hostname(); net.ParseIP(host) == nil && !strings.EqualFold(host, "localhost") && !strings.EqualFold(host, name) {
		return false
	}

	peer, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return false
	}
	ip := net.ParseIP(peer)
	if ip == nil {
		return false
	}
	if ip.IsLoopback() {
		return true
	}
	for _, a := range strings.Split(os.Getenv("OLLAMA_CLUSTER_ADMIN_ADDRS"), ",") {
		if allowed := net.ParseIP(strings.TrimSpace(a)); allowed != nil && allowed.Equal(ip) {
			return true
		}
	}
	return false
}
