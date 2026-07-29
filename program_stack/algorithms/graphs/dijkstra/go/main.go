package main

import (
	"fmt"
	"os"
)

const INF = 1_000_000_000

// Dijkstra O(V^2). edges: {u, v, w}. Unreachable = INF.
func Dijkstra(n int, edges [][3]int, source int) []int {
	w := make([][]int, n)
	for i := range w {
		w[i] = make([]int, n)
		for j := range w[i] {
			w[i][j] = -1
		}
	}
	for _, e := range edges {
		w[e[0]][e[1]] = e[2]
	}
	dist := make([]int, n)
	done := make([]bool, n)
	for i := range dist {
		dist[i] = INF
	}
	dist[source] = 0

	for iter := 0; iter < n; iter++ {
		u := -1
		for i := 0; i < n; i++ {
			if !done[i] && (u < 0 || dist[i] < dist[u]) {
				u = i
			}
		}
		if u < 0 || dist[u] == INF {
			break
		}
		done[u] = true
		for v := 0; v < n; v++ {
			if w[u][v] >= 0 && dist[u]+w[u][v] < dist[v] {
				dist[v] = dist[u] + w[u][v]
			}
		}
	}
	return dist
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
	edges := [][3]int{{0, 1, 1}, {0, 2, 4}, {0, 3, 2}, {1, 2, 1}, {3, 2, 1}}
	dist := Dijkstra(4, edges, 0)
	if !eq(dist, []int{0, 1, 2, 2}) {
		fmt.Fprintf(os.Stderr, "unexpected dist: %v\n", dist)
		os.Exit(1)
	}
	dist = Dijkstra(3, [][3]int{{0, 1, 5}}, 0)
	if dist[0] != 0 || dist[1] != 5 || dist[2] != INF {
		fmt.Fprintf(os.Stderr, "unexpected unreachable case: %v\n", dist)
		os.Exit(1)
	}
	fmt.Println("dijkstra: ok")
}
