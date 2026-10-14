package cluster

import (
	"fmt"
	"net"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
)

// The worker fields end up in SSH command lines, shell scripts and systemd
// units on the worker, so they are held to plain forms: IP literals, a
// user@host target, simple names. The config can be written through the API,
// so this is checked on every save and before any command uses it.
var (
	sshTarget = regexp.MustCompile(`^[A-Za-z0-9._][A-Za-z0-9._-]*@[A-Za-z0-9.:_-]+$`)
	plainName = regexp.MustCompile(`^[A-Za-z0-9._-]*$`)
	imageRef  = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._/:@-]*$`)
)

// Validate checks that the worker fields are safe to use in commands.
func (w Worker) Validate() error {
	label := w.Name
	if label == "" {
		label = w.Addr
	}
	if !plainName.MatchString(w.Name) {
		return fmt.Errorf("worker %q: the name may contain only letters, digits, '.', '_' and '-'", label)
	}
	if w.SSH != "" && !sshTarget.MatchString(w.SSH) {
		return fmt.Errorf("worker %s: ssh must be user@host, got %q", label, w.SSH)
	}
	if ip := net.ParseIP(w.Addr); ip == nil || ip.IsUnspecified() || ip.IsMulticast() {
		return fmt.Errorf("worker %s: addr must be the IP address of the worker, got %q", label, w.Addr)
	}
	if w.LocalAddr != "" && net.ParseIP(w.LocalAddr) == nil {
		return fmt.Errorf("worker %s: local_addr must be an IP address, got %q", label, w.LocalAddr)
	}
	for _, p := range []int{w.RPCPort, w.StatusPort} {
		if p < 0 || p > 65535 {
			return fmt.Errorf("worker %s: port %d is out of range", label, p)
		}
	}
	switch w.Mode {
	case "", "docker", "native":
	default:
		return fmt.Errorf("worker %s: mode must be docker or native, got %q", label, w.Mode)
	}
	if w.Image != "" && !imageRef.MatchString(w.Image) {
		return fmt.Errorf("worker %s: %q is not a Docker image name", label, w.Image)
	}
	return nil
}

// A location is mounted into the Ollama container read-write, so system
// directories are refused: these ones with everything below them, and
// systemRoots themselves.
var (
	systemTrees = []string{"/bin", "/boot", "/dev", "/etc", "/lib", "/lib64", "/proc", "/root/.ssh", "/sbin", "/sys", "/usr", "/var/lib/docker", "/var/run"}
	systemRoots = []string{"/", "/home", "/root", "/run", "/var", "/var/lib"}
)

// validateLocations checks that locations are absolute, clean paths that can
// be mounted into a container.
func validateLocations(locs []Location) error {
	for _, l := range locs {
		if !filepath.IsAbs(l.Path) || filepath.Clean(l.Path) != l.Path || strings.ContainsAny(l.Path, ":,\n") {
			return fmt.Errorf("location %q must be an absolute path without '..', ':', ',' or a trailing '/'", l.Path)
		}
		if slices.Contains(systemRoots, l.Path) || slices.ContainsFunc(systemTrees, func(d string) bool { return l.Path == d || strings.HasPrefix(l.Path, d+"/") }) {
			return fmt.Errorf("location %q is a system directory", l.Path)
		}
	}
	return nil
}
