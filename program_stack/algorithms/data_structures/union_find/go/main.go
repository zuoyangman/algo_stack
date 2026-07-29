package main

import (
	"fmt"
	"os"
)

type UnionFind struct {
	parent []int
	rank   []int
}

func NewUnionFind(n int) *UnionFind {
	parent := make([]int, n)
	for i := range parent {
		parent[i] = i
	}
	return &UnionFind{parent: parent, rank: make([]int, n)}
}

func (uf *UnionFind) Find(x int) int {
	if uf.parent[x] != x {
		uf.parent[x] = uf.Find(uf.parent[x])
	}
	return uf.parent[x]
}

func (uf *UnionFind) Union(a, b int) bool {
	ra := uf.Find(a)
	rb := uf.Find(b)
	if ra == rb {
		return false
	}
	if uf.rank[ra] < uf.rank[rb] {
		uf.parent[ra] = rb
	} else if uf.rank[ra] > uf.rank[rb] {
		uf.parent[rb] = ra
	} else {
		uf.parent[rb] = ra
		uf.rank[ra]++
	}
	return true
}

func (uf *UnionFind) Connected(a, b int) bool {
	return uf.Find(a) == uf.Find(b)
}

func expect(cond bool, msg string) {
	if !cond {
		fmt.Fprintln(os.Stderr, "FAIL:", msg)
		os.Exit(1)
	}
}

func main() {
	uf := NewUnionFind(6)
	expect(!uf.Connected(0, 1), "initially separate")
	expect(uf.Union(0, 1), "union 0-1")
	expect(uf.Union(1, 2), "union 1-2")
	expect(uf.Connected(0, 2), "0 connected 2")
	expect(!uf.Connected(0, 3), "0 not 3")
	expect(uf.Union(3, 4), "union 3-4")
	expect(uf.Union(2, 4), "merge components")
	expect(uf.Connected(0, 3), "0 connected 3")
	expect(!uf.Union(0, 2), "already united")
	expect(uf.Find(5) == 5, "singleton 5")
	fmt.Println("union_find: ok")
}
