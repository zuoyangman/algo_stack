#include <algorithm>
#include <iostream>
#include <vector>

// Classic 0/1 knapsack: maximum value with capacity W.
int knapsack01(const std::vector<int>& weights, const std::vector<int>& values, int capacity) {
    std::vector<int> dp(capacity + 1, 0);
    for (size_t i = 0; i < weights.size(); ++i) {
        int w = weights[i], v = values[i];
        for (int j = capacity; j >= w; --j) {
            dp[j] = std::max(dp[j], dp[j - w] + v);
        }
    }
    return dp[capacity];
}

int main() {
    if (knapsack01({2, 3, 4, 5}, {3, 4, 5, 6}, 5) != 7) {
        std::cerr << "expected 7\n";
        return 1;
    }
    if (knapsack01({1, 2, 3}, {6, 10, 12}, 5) != 22) {
        std::cerr << "expected 22\n";
        return 1;
    }
    if (knapsack01({}, {}, 10) != 0) {
        std::cerr << "empty items should be 0\n";
        return 1;
    }
    std::cout << "knapsack_01: ok\n";
    return 0;
}
