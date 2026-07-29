package main

import (
	"fmt"
	"os"
)

type node struct {
	key   int
	left  *node
	right *node
}

type BST struct {
	root *node
}

func (t *BST) Insert(x int) {
	t.root = insertRec(t.root, x)
}

func insertRec(n *node, x int) *node {
	if n == nil {
		return &node{key: x}
	}
	if x < n.key {
		n.left = insertRec(n.left, x)
	} else if x > n.key {
		n.right = insertRec(n.right, x)
	}
	return n
}

func (t *BST) Contains(x int) bool {
	cur := t.root
	for cur != nil {
		if x == cur.key {
			return true
		}
		if x < cur.key {
			cur = cur.left
		} else {
			cur = cur.right
		}
	}
	return false
}

func (t *BST) Inorder() []int {
	var out []int
	var walk func(*node)
	walk = func(n *node) {
		if n == nil {
			return
		}
		walk(n.left)
		out = append(out, n.key)
		walk(n.right)
	}
	walk(t.root)
	return out
}

func expect(cond bool, msg string) {
	if !cond {
		fmt.Fprintln(os.Stderr, "FAIL:", msg)
		os.Exit(1)
	}
}

func equal(a, b []int) bool {
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
	t := &BST{}
	for _, v := range []int{5, 3, 7, 1, 4, 6, 8} {
		t.Insert(v)
	}
	t.Insert(5)
	expect(t.Contains(4), "contains 4")
	expect(!t.Contains(2), "missing 2")
	expect(equal(t.Inorder(), []int{1, 3, 4, 5, 6, 7, 8}), "inorder")
	fmt.Println("binary_search_tree: ok")
}
