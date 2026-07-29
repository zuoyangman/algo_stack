package main

import (
	"fmt"
	"os"
)

// BuildLps returns the LPS / failure function for pattern.
func BuildLps(pattern string) []int {
	m := len(pattern)
	lps := make([]int, m)
	lenBorder, i := 0, 1
	for i < m {
		if pattern[i] == pattern[lenBorder] {
			lenBorder++
			lps[i] = lenBorder
			i++
		} else if lenBorder > 0 {
			lenBorder = lps[lenBorder-1]
		} else {
			lps[i] = 0
			i++
		}
	}
	return lps
}

// KmpSearch returns all starting indices of pattern in text.
func KmpSearch(text, pattern string) []int {
	hits := []int{}
	if len(pattern) == 0 {
		return hits
	}
	lps := BuildLps(pattern)
	n, m := len(text), len(pattern)
	i, q := 0, 0
	for i < n {
		if text[i] == pattern[q] {
			i++
			q++
			if q == m {
				hits = append(hits, i-m)
				q = lps[q-1]
			}
		} else if q > 0 {
			q = lps[q-1]
		} else {
			i++
		}
	}
	return hits
}

// KmpFirst returns the first match index, or -1.
func KmpFirst(text, pattern string) int {
	hits := KmpSearch(text, pattern)
	if len(hits) == 0 {
		return -1
	}
	return hits[0]
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
	hits := KmpSearch("ababcabab", "abab")
	if !eq(hits, []int{0, 5}) {
		fmt.Fprintf(os.Stderr, "unexpected hits: %v\n", hits)
		os.Exit(1)
	}
	if KmpFirst("ababcabab", "abab") != 0 {
		fmt.Fprintln(os.Stderr, "first index failed")
		os.Exit(1)
	}
	if !eq(KmpSearch("aaaa", "aa"), []int{0, 1, 2}) {
		fmt.Fprintln(os.Stderr, "overlap hits failed")
		os.Exit(1)
	}
	if KmpFirst("hello", "world") != -1 {
		fmt.Fprintln(os.Stderr, "miss should be -1")
		os.Exit(1)
	}
	fmt.Println("kmp: ok")
}
