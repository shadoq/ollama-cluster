package clustercmd

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// remote runs commands on the worker machine over SSH. Authentication is the
// user's own SSH setup (keys, agent, config); when no key works, `add` copies
// one with ssh-copy-id so the password is typed once.
type remote struct {
	target string // user@host or host
}

func (r remote) host() string {
	if _, h, ok := strings.Cut(r.target, "@"); ok {
		return h
	}
	return r.target
}

func (r remote) user() string {
	if u, _, ok := strings.Cut(r.target, "@"); ok {
		return u
	}
	return ""
}

func sshArgs(target string, batch bool) []string {
	args := []string{"-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=15", "-o", "StrictHostKeyChecking=accept-new"}
	if batch {
		args = append(args, "-o", "BatchMode=yes")
	}
	// "--": a target can never be read as an ssh option
	return append(args, "--", target)
}

// run executes a shell script on the worker and returns its stdout.
func (r remote) run(ctx context.Context, script string) (string, error) {
	cmd := exec.CommandContext(ctx, "ssh", append(sshArgs(r.target, true), "sh", "-s")...)
	cmd.Stdin = strings.NewReader(script)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = err.Error()
		}
		return stdout.String(), errors.New(msg)
	}
	return stdout.String(), nil
}

// runTTY executes a command on the worker with the local terminal attached,
// for steps that ask for the sudo password.
func (r remote) runTTY(ctx context.Context, command string) error {
	cmd := exec.CommandContext(ctx, "ssh", append([]string{"-t"}, append(sshArgs(r.target, false), command)...)...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	return cmd.Run()
}

// ensureAccess checks key-based SSH access and, in a terminal, installs the
// user's key on the worker when only a password works.
func (r remote) ensureAccess(ctx context.Context) error {
	if strings.HasPrefix(r.target, "-") {
		return fmt.Errorf("%q is not an SSH target (user@host)", r.target)
	}
	if _, err := r.run(ctx, "true"); err == nil {
		return nil
	} else if !strings.Contains(err.Error(), "Permission denied") {
		return fmt.Errorf("cannot connect to %s over SSH: %v", r.target, err)
	}

	if !interactive() {
		return fmt.Errorf("SSH to %s needs a password; run 'ssh-copy-id %s' once and try again", r.target, r.target)
	}
	fmt.Printf("  %s accepts only a password. Copying your SSH key there (asked once).\n", r.host())
	cmd := exec.CommandContext(ctx, "ssh-copy-id", r.target)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("ssh-copy-id %s: %w", r.target, err)
	}
	_, err := r.run(ctx, "true")
	return err
}

// pipeTo streams the output of a local command into a command on the worker,
// e.g. `docker save | ssh worker docker load`.
func (r remote) pipeTo(ctx context.Context, local *exec.Cmd, remoteCmd string) error {
	ssh := exec.CommandContext(ctx, "ssh", append(sshArgs(r.target, true), remoteCmd)...)
	pipe, err := local.StdoutPipe()
	if err != nil {
		return err
	}
	ssh.Stdin = pipe
	var lstderr, rstderr bytes.Buffer
	local.Stderr = &lstderr
	ssh.Stderr = &rstderr
	ssh.Stdout = &rstderr

	if err := ssh.Start(); err != nil {
		return err
	}
	if err := local.Run(); err != nil {
		_ = ssh.Wait()
		return fmt.Errorf("%s: %v %s", local.Path, err, strings.TrimSpace(lstderr.String()))
	}
	if err := ssh.Wait(); err != nil {
		return fmt.Errorf("%s on %s: %v %s", remoteCmd, r.host(), err, strings.TrimSpace(rstderr.String()))
	}
	return nil
}

// netIf is a network interface with an IPv4 address.
type netIf struct {
	Name    string
	Addr    net.IP
	Net     *net.IPNet
	RDMADev string // e.g. rocep1s0f0 when the interface supports RDMA
}

// localLinkInterfaces lists local interfaces that support RDMA, which on a
// DGX Spark are the ConnectX-7 ports of the direct cable.
func localLinkInterfaces() []netIf {
	ifaces, err := net.Interfaces()
	if err != nil {
		return nil
	}
	var out []netIf
	for _, iface := range ifaces {
		if iface.Flags&net.FlagUp == 0 || iface.Flags&net.FlagLoopback != 0 {
			continue
		}
		addrs, _ := iface.Addrs()
		for _, a := range addrs {
			ip, ipnet, err := net.ParseCIDR(a.String())
			if err != nil || ip.To4() == nil {
				continue
			}
			out = append(out, netIf{Name: iface.Name, Addr: ip, Net: ipnet, RDMADev: rdmaDevice(iface.Name)})
		}
	}
	return out
}

func rdmaDevice(iface string) string {
	entries, err := os.ReadDir(filepath.Join("/sys/class/net", iface, "device", "infiniband"))
	if err != nil || len(entries) == 0 {
		return ""
	}
	return entries[0].Name()
}

// remoteInterfacesScript prints "name addr/prefix rdma-device" per IPv4
// address on the worker.
const remoteInterfacesScript = `
ip -o -4 addr show up 2>/dev/null | while read -r idx name fam addr rest; do
  dev=$(ls /sys/class/net/"$name"/device/infiniband 2>/dev/null | head -1)
  echo "$name $addr ${dev:--}"
done
`

func parseRemoteInterfaces(out string) []netIf {
	var ifs []netIf
	scanner := bufio.NewScanner(strings.NewReader(out))
	for scanner.Scan() {
		f := strings.Fields(scanner.Text())
		if len(f) < 3 {
			continue
		}
		ip, ipnet, err := net.ParseCIDR(f[1])
		if err != nil {
			continue
		}
		dev := f[2]
		if dev == "-" {
			dev = ""
		}
		ifs = append(ifs, netIf{Name: f[0], Addr: ip, Net: ipnet, RDMADev: dev})
	}
	return ifs
}

// linkPair is the direct link between this machine and the worker.
type linkPair struct {
	Local  netIf
	Remote netIf
}

// findLink matches local and remote interfaces on the same subnet. Only RDMA
// interfaces on both ends count as the direct cable; a shared LAN subnet is
// accepted only with allowLAN. wantAddr selects the remote address.
func findLink(local, remote []netIf, wantAddr string, allowLAN bool) (linkPair, bool) {
	var rdma, lan []linkPair
	for _, r := range remote {
		if wantAddr != "" && r.Addr.String() != wantAddr {
			continue
		}
		for _, l := range local {
			if !l.Net.Contains(r.Addr) || l.Addr.Equal(r.Addr) {
				continue
			}
			p := linkPair{Local: l, Remote: r}
			if l.RDMADev != "" && r.RDMADev != "" {
				rdma = append(rdma, p)
			} else {
				lan = append(lan, p)
			}
		}
	}
	switch {
	case len(rdma) > 0:
		return rdma[0], true
	case allowLAN && len(lan) > 0:
		return lan[0], true
	}
	return linkPair{}, false
}

func ping(ctx context.Context, addr string) error {
	return exec.CommandContext(ctx, "ping", "-c", "1", "-W", "2", addr).Run()
}
