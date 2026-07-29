package main

import (
	"fmt"
	"os"
)

// Bfs returns the visit order from start on an adjacency-list graph.
func Bfs(adj [][]int, start int) []int {
	n := len(adj)
	visited := make([]bool, n)
	order := make([]int, 0, n)
	q := []int{start}
	visited[start] = true
	for len(q) > 0 {
		u := q[0]
		q = q[1:]
		order = append(order, u)
		for _, v := range adj[u] {
			if !visited[v] {
				visited[v] = true
				q = append(q, v)
			}
		}
	}
	return order
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
	// 0 → 1,2 ; 1 → 3 ; 2 → 3 ; 3 → 4
	adj := [][]int{{1, 2}, {3}, {3}, {4}, {}}
	order := Bfs(adj, 0)
	if !eq(order, []int{0, 1, 2, 3, 4}) {
		fmt.Fprintf(os.Stderr, "unexpected order: %v\n", order)
		os.Exit(1)
	}
	order = Bfs(adj, 3)
	if !eq(order, []int{3, 4}) {
		fmt.Fprintf(os.Stderr, "unexpected order from 3: %v\n", order)
		os.Exit(1)
	}
	fmt.Println("bfs: ok")
}
