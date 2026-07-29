#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

int lcsLength(const std::string& a, const std::string& b) {
    int n = static_cast<int>(a.size()), m = static_cast<int>(b.size());
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            if (a[i - 1] == b[j - 1]) dp[i][j] = dp[i - 1][j - 1] + 1;
            else dp[i][j] = std::max(dp[i - 1][j], dp[i][j - 1]);
        }
    }
    return dp[n][m];
}

std::string lcsString(const std::string& a, const std::string& b) {
    int n = static_cast<int>(a.size()), m = static_cast<int>(b.size());
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            if (a[i - 1] == b[j - 1]) dp[i][j] = dp[i - 1][j - 1] + 1;
            else dp[i][j] = std::max(dp[i - 1][j], dp[i][j - 1]);
        }
    }
    std::string s;
    int i = n, j = m;
    while (i > 0 && j > 0) {
        if (a[i - 1] == b[j - 1]) {
            s.push_back(a[i - 1]);
            --i;
            --j;
        } else if (dp[i - 1][j] >= dp[i][j - 1]) {
            --i;
        } else {
            --j;
        }
    }
    std::reverse(s.begin(), s.end());
    return s;
}

int main() {
    std::string a = "ABCBDAB", b = "BDCABA";
    if (lcsLength(a, b) != 4) {
        std::cerr << "expected length 4\n";
        return 1;
    }
    std::string s = lcsString(a, b);
    if (static_cast<int>(s.size()) != 4 || lcsLength(s, a) != 4 || lcsLength(s, b) != 4) {
        std::cerr << "bad LCS string\n";
        return 1;
    }
    if (lcsLength("", "xyz") != 0 || lcsLength("abc", "abc") != 3) {
        std::cerr << "edge cases failed\n";
        return 1;
    }
    std::cout << "lcs: ok\n";
    return 0;
}
