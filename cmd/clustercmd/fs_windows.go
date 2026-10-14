package clustercmd

import "golang.org/x/sys/windows"

func diskFree(path string) (uint64, bool) {
	p, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return 0, false
	}
	var free uint64
	if err := windows.GetDiskFreeSpaceEx(p, &free, nil, nil); err != nil {
		return 0, false
	}
	return free, true
}

func sameDisk(a, b string) bool {
	return len(a) >= 2 && len(b) >= 2 && a[:2] == b[:2]
}

func fileOwner(string) (int, int, bool) {
	return 0, 0, false
}
