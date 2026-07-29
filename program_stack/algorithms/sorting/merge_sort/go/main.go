package main

import (
	"fmt"
	"os"
)

func merge(a []int, lo, mid, hi int, buf []int) {
	i, j, k := lo, mid, lo
	for i < mid && j < hi {
		if a[i] <= a[j] {
			buf[k] = a[i]
			i++
		} else {
			buf[k] = a[j]
			j++
		}
		k++
	}
	for i < mid {
		buf[k] = a[i]
		i++
		k++
	}
	for j < hi {
		buf[k] = a[j]
		j++
		k++
	}
	for t := lo; t < hi; t++ {
		a[t] = buf[t]
	}
}

func sortRange(a []int, lo, hi int, buf []int) {
	if hi-lo <= 1 {
		return
	}
	mid := lo + (hi-lo)/2
	sortRange(a, lo, mid, buf)
	sortRange(a, mid, hi, buf)
	merge(a, lo, mid, hi, buf)
}

// Sort sorts a in non-decreasing order (top-down merge sort). Stable.
func Sort(a []int) {
	if len(a) < 2 {
		return
	}
	buf := make([]int, len(a))
	sortRange(a, 0, len(a), buf)
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
	fmt.Println("merge_sort: ok")
}
