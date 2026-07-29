package main

import (
	"fmt"
	"os"
)

// Search returns an index of x in sorted a, or -1 if absent.
func Search(a []int, x int) int {
	if len(a) == 0 {
		return -1
	}
	lo, hi := 0, len(a)-1
	for lo <= hi {
		mid := lo + (hi-lo)/2
		if a[mid] == x {
			return mid
		} else if a[mid] < x {
			lo = mid + 1
		} else {
			hi = mid - 1
		}
	}
	return -1
}

func expectExact(a []int, x, want int) {
	got := Search(a, x)
	if got != want {
		fmt.Fprintf(os.Stderr, "search got %d want %d\n", got, want)
		os.Exit(1)
	}
}

func expectHit(a []int, x int) {
	got := Search(a, x)
	if got < 0 || got >= len(a) || a[got] != x {
		fmt.Fprintf(os.Stderr, "expected hit for %d\n", x)
		os.Exit(1)
	}
}

func main() {
	expectExact([]int{}, 1, -1)
	expectExact([]int{42}, 42, 0)
	expectExact([]int{42}, 7, -1)
	expectHit([]int{1, 2, 3, 4, 5}, 1)
	expectHit([]int{1, 2, 3, 4, 5}, 5)
	expectHit([]int{1, 2, 3, 4, 5}, 3)
	expectExact([]int{1, 2, 3, 4, 5}, 6, -1)
	expectHit([]int{1, 1, 2, 2, 3}, 2)
	fmt.Println("binary_search: ok")
}
