package main

import (
	"fmt"
	"os"
)

// Search returns the index of the first x in a, or -1 if absent.
func Search(a []int, x int) int {
	for i, v := range a {
		if v == x {
			return i
		}
	}
	return -1
}

func expect(a []int, x, want int) {
	got := Search(a, x)
	if got != want {
		fmt.Fprintf(os.Stderr, "search got %d want %d\n", got, want)
		os.Exit(1)
	}
}

func main() {
	expect([]int{}, 1, -1)
	expect([]int{42}, 42, 0)
	expect([]int{42}, 7, -1)
	expect([]int{3, 1, 4, 1, 5}, 1, 1)
	expect([]int{3, 1, 4, 1, 5}, 5, 4)
	expect([]int{3, 1, 4, 1, 5}, 9, -1)
	fmt.Println("linear_search: ok")
}
