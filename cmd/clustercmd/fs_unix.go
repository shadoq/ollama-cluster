//go:build !windows

package clustercmd

import (
	"os"
	"syscall"
)

func diskFree(path string) (uint64, bool) {
	var s syscall.Statfs_t
	if err := syscall.Statfs(path, &s); err != nil {
		return 0, false
	}
	return uint64(s.Bavail) * uint64(s.Bsize), true //nolint:unconvert // types differ per OS
}

// sameDisk reports whether two existing paths are on one filesystem, so a
// file can be moved between them with a rename.
func sameDisk(a, b string) bool {
	fa, err := os.Stat(a)
	if err != nil {
		return false
	}
	fb, err := os.Stat(b)
	if err != nil {
		return false
	}
	sa, ok1 := fa.Sys().(*syscall.Stat_t)
	sb, ok2 := fb.Sys().(*syscall.Stat_t)
	return ok1 && ok2 && uint64(sa.Dev) == uint64(sb.Dev) //nolint:unconvert
}

func fileOwner(path string) (uid, gid int, ok bool) {
	fi, err := os.Stat(path)
	if err != nil {
		return 0, 0, false
	}
	st, ok := fi.Sys().(*syscall.Stat_t)
	if !ok {
		return 0, 0, false
	}
	return int(st.Uid), int(st.Gid), true
}
