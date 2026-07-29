package main

import (
	"fmt"
	"os"
)

func siftDown(a []int, heapSize, i int) {
	for {
		largest := i
		left := 2*i + 1
		right := 2*i + 2
		if left < heapSize && a[left] > a[largest] {
			largest = left
		}
		if right < heapSize && a[right] > a[largest] {
			largest = right
		}
		if largest == i {
			return
		}
		a[i], a[largest] = a[largest], a[i]
		i = largest
	}
}

// Sort sorts a in non-decreasing order in place (heapsort). Not stable.
func Sort(a []int) {
	if len(a) < 2 {
		return
	}
	n := len(a)
	for i := n/2 - 1; i >= 0; i-- {
		siftDown(a, n, i)
	}
	for end := n - 1; end > 0; end-- {
		a[0], a[end] = a[end], a[0]
		siftDown(a, end, 0)
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
	fmt.Println("heap_sort: ok")
}
