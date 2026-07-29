package main

import (
	"fmt"
	"os"
)

// TopologicalSort is Kahn's algorithm. ok=false if the graph has a cycle.
func TopologicalSort(adj [][]int) (order []int, ok bool) {
	n := len(adj)
	indeg := make([]int, n)
	for u := 0; u < n; u++ {
		for _, v := range adj[u] {
			indeg[v]++
		}
	}
	q := make([]int, 0)
	for i := 0; i < n; i++ {
		if indeg[i] == 0 {
			q = append(q, i)
		}
	}
	order = make([]int, 0, n)
	for len(q) > 0 {
		u := q[0]
		q = q[1:]
		order = append(order, u)
		for _, v := range adj[u] {
			indeg[v]--
			if indeg[v] == 0 {
				q = append(q, v)
			}
		}
	}
	return order, len(order) == n
}

func eq(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func main() {
	adj := [][]int{{1, 2}, {3}, {3}, {}}
	order, ok := TopologicalSort(adj)
	if !ok || !eq(order, []int{0, 1, 2, 3}) {
		fmt.Fprintf(os.Stderr, "unexpected order: %v ok=%v\n", order, ok)
		os.Exit(1)
	}
	_, ok = TopologicalSort([][]int{{1}, {0}})
	if ok {
		fmt.Fprintln(os.Stderr, "expected cycle detection failure")
		os.Exit(1)
	}
	fmt.Println("topological_sort: ok")
}
