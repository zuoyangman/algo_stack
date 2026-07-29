package main

import (
	"fmt"
	"os"
)

// LcsLength returns the length of an LCS of a and b.
func LcsLength(a, b string) int {
	n, m := len(a), len(b)
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, m+1)
	}
	for i := 1; i <= n; i++ {
		for j := 1; j <= m; j++ {
			if a[i-1] == b[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else if dp[i-1][j] >= dp[i][j-1] {
				dp[i][j] = dp[i-1][j]
			} else {
				dp[i][j] = dp[i][j-1]
			}
		}
	}
	return dp[n][m]
}

// LcsString returns one concrete LCS string.
func LcsString(a, b string) string {
	n, m := len(a), len(b)
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, m+1)
	}
	for i := 1; i <= n; i++ {
		for j := 1; j <= m; j++ {
			if a[i-1] == b[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else if dp[i-1][j] >= dp[i][j-1] {
				dp[i][j] = dp[i-1][j]
			} else {
				dp[i][j] = dp[i][j-1]
			}
		}
	}
	out := make([]byte, 0, dp[n][m])
	i, j := n, m
	for i > 0 && j > 0 {
		if a[i-1] == b[j-1] {
			out = append(out, a[i-1])
			i--
			j--
		} else if dp[i-1][j] >= dp[i][j-1] {
			i--
		} else {
			j--
		}
	}
	for l, r := 0, len(out)-1; l < r; l, r = l+1, r-1 {
		out[l], out[r] = out[r], out[l]
	}
	return string(out)
}

func main() {
	a, b := "ABCBDAB", "BDCABA"
	if LcsLength(a, b) != 4 {
		fmt.Fprintln(os.Stderr, "expected length 4")
		os.Exit(1)
	}
	s := LcsString(a, b)
	if len(s) != 4 || LcsLength(s, a) != 4 || LcsLength(s, b) != 4 {
		fmt.Fprintf(os.Stderr, "bad LCS string: %q\n", s)
		os.Exit(1)
	}
	if LcsLength("", "xyz") != 0 || LcsLength("abc", "abc") != 3 {
		fmt.Fprintln(os.Stderr, "edge cases failed")
		os.Exit(1)
	}
	fmt.Println("lcs: ok")
}
