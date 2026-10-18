package worker

import "testing"

func TestParseProcNetTCP(t *testing.T) {
	data := `  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 0230640A:C344 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 1 1 0000000000000000 100 0 0 10 0
   1: 0230640A:C344 0130640A:D2F0 01 00000000:00000000 00:00000000 00000000     0        0 2 1 0000000000000000 20 4 30 10 -1
   2: 0230640A:C345 0130640A:D2F1 01 00000000:00000000 00:00000000 00000000     0        0 3 1 0000000000000000 20 4 30 10 -1
`
	listening, clients := parseProcNetTCP(data, ":C344")
	if !listening || clients != 1 {
		t.Fatalf("got listening=%v clients=%d, want true 1", listening, clients)
	}
	listening, clients = parseProcNetTCP(data, ":0050")
	if listening || clients != 0 {
		t.Fatalf("other port: got listening=%v clients=%d", listening, clients)
	}
}

func TestLinkState(t *testing.T) {
	var l linkState
	steps := []struct {
		line string
		want string
	}{
		{"  transport      : TCP (RDMA auto-negotiate enabled)", ""},
		{"Accepted client connection", "TCP"},
		{"RDMA probed: dev=rocep1s0f0 gid=3 RoCEv2 qpn=4812 inline=316", "TCP"},
		{"RDMA activated: qpn=4812->4927 mtu=1024 rx_depth=24", "RDMA (RoCEv2, rocep1s0f0, MTU 1024)"},
		{"Client connection closed", "RDMA (RoCEv2, rocep1s0f0, MTU 1024)"},
		{"Accepted client connection", "TCP"},
		{"RDMA activate failed, staying on TCP", "TCP (RDMA activation failed)"},
	}
	for _, s := range steps {
		l.update(s.line)
		if l.label != s.want {
			t.Fatalf("after %q: got %q, want %q", s.line, l.label, s.want)
		}
	}
}

func TestParseDevices(t *testing.T) {
	out := `Available devices:
  CUDA0: NVIDIA GB10 (124546 MiB, 110000 MiB free)
  CPU: ARM Cortex (124546 MiB, 110000 MiB free)
`
	devices := parseDevices(out)
	if len(devices) != 1 || devices[0].Name != "CUDA0" || devices[0].TotalMemory != 124546<<20 {
		t.Fatalf("unexpected devices %+v", devices)
	}
}
