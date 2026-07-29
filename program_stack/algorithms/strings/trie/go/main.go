package main

import (
	"fmt"
	"os"
)

type node struct {
	next map[rune]*node
	end  bool
}

// Trie supports insert / search / startsWith.
type Trie struct {
	root *node
}

func NewTrie() *Trie {
	return &Trie{root: &node{next: make(map[rune]*node)}}
}

func (t *Trie) Insert(word string) {
	cur := t.root
	for _, c := range word {
		if cur.next[c] == nil {
			cur.next[c] = &node{next: make(map[rune]*node)}
		}
		cur = cur.next[c]
	}
	cur.end = true
}

func (t *Trie) Search(word string) bool {
	cur := t.walk(word)
	return cur != nil && cur.end
}

func (t *Trie) StartsWith(prefix string) bool {
	return t.walk(prefix) != nil
}

func (t *Trie) walk(s string) *node {
	cur := t.root
	for _, c := range s {
		cur = cur.next[c]
		if cur == nil {
			return nil
		}
	}
	return cur
}

func main() {
	t := NewTrie()
	t.Insert("apple")
	if !t.Search("apple") || t.Search("app") || !t.StartsWith("app") {
		fmt.Fprintln(os.Stderr, "apple/app checks failed")
		os.Exit(1)
	}
	t.Insert("app")
	if !t.Search("app") || !t.StartsWith("ap") || t.Search("appl") {
		fmt.Fprintln(os.Stderr, "after insert app failed")
		os.Exit(1)
	}
	if t.Search("") || t.StartsWith("b") {
		fmt.Fprintln(os.Stderr, "edge checks failed")
		os.Exit(1)
	}
	fmt.Println("trie: ok")
}
