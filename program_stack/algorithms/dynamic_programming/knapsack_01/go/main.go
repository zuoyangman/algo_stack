package main

import (
	"fmt"
	"os"
)

// Knapsack01 returns the maximum value for capacity W (0/1).
func Knapsack01(weights, values []int, capacity int) int {
	dp := make([]int, capacity+1)
	for i := range weights {
		w, v := weights[i], values[i]
		for j := capacity; j >= w; j-- {
			if dp[j-w]+v > dp[j] {
				dp[j] = dp[j-w] + v
			}
		}
	}
	return dp[capacity]
}

func main() {
	if Knapsack01([]int{2, 3, 4, 5}, []int{3, 4, 5, 6}, 5) != 7 {
		fmt.Fprintln(os.Stderr, "expected 7")
		os.Exit(1)
	}
	if Knapsack01([]int{1, 2, 3}, []int{6, 10, 12}, 5) != 22 {
		fmt.Fprintln(os.Stderr, "expected 22")
		os.Exit(1)
	}
	if Knapsack01(nil, nil, 10) != 0 {
		fmt.Fprintln(os.Stderr, "empty items should be 0")
		os.Exit(1)
	}
	fmt.Println("knapsack_01: ok")
}
