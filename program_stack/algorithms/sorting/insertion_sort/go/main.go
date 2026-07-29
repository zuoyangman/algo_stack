package main

import (
	"fmt"
	"os"
)

// Sort sorts a in non-decreasing order in place (insertion sort). Stable.
func Sort(a []int) {
	for i := 1; i < len(a); i++ {
		key := a[i]
		j := i
		for j > 0 && a[j-1] > key {
			a[j] = a[j-1]
			j--
		}
		a[j] = key
	}
}

func expect(input, want []int) {
	a := append([]int(nil), input...)
	Sort(a)
	if len(a) != len(want) {
		fmt.Fprintln(os.Stderr, "length mismatch")
		os.Exit(1)
	}
	for i := range a {
		if a[i] != want[i] {
			fmt.Fprintln(os.Stderr, "value mismatch")
			os.Exit(1)
		}
	}
}

func main() {
	expect([]int{}, []int{})
	expect([]int{42}, []int{42})
	expect([]int{1, 2, 3}, []int{1, 2, 3})
	expect([]int{3, 2, 1}, []int{1, 2, 3})
	expect([]int{5, 1, 4, 2, 8}, []int{1, 2, 4, 5, 8})
	expect([]int{3, 1, 2, 1, 3}, []int{1, 1, 2, 3, 3})
	fmt.Println("insertion_sort: ok")
}
