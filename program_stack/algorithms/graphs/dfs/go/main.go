package main

import (
	"fmt"
	"os"
)

// Dfs returns recursive DFS visit order (preorder) from start.
func Dfs(adj [][]int, start int) []int {
	visited := make([]bool, len(adj))
	order := make([]int, 0, len(adj))
	dfsRec(adj, start, visited, &order)
	return order
}

func dfsRec(adj [][]int, u int, visited []bool, order *[]int) {
	visited[u] = true
	*order = append(*order, u)
	for _, v := range adj[u] {
		if !visited[v] {
			dfsRec(adj, v, visited, order)
		}
	}
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
	// 0 → 1,2 ; 1 → 3 ; 2 → 3 ; 3 → 4  → order 0,1,3,4,2
	adj := [][]int{{1, 2}, {3}, {3}, {4}, {}}
	order := Dfs(adj, 0)
	if !eq(order, []int{0, 1, 3, 4, 2}) {
		fmt.Fprintf(os.Stderr, "unexpected order: %v\n", order)
		os.Exit(1)
	}
	order = Dfs(adj, 2)
	if !eq(order, []int{2, 3, 4}) {
		fmt.Fprintf(os.Stderr, "unexpected order from 2: %v\n", order)
		os.Exit(1)
	}
	fmt.Println("dfs: ok")
}
