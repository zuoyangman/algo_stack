#include <iostream>
#include <string>
#include <vector>

std::vector<int> buildLps(const std::string& pattern) {
    int m = static_cast<int>(pattern.size());
    std::vector<int> lps(m, 0);
    int len = 0, i = 1;
    while (i < m) {
        if (pattern[i] == pattern[len]) {
            lps[i++] = ++len;
        } else if (len > 0) {
            len = lps[len - 1];
        } else {
            lps[i++] = 0;
        }
    }
    return lps;
}

std::vector<int> kmpSearch(const std::string& text, const std::string& pattern) {
    std::vector<int> hits;
    if (pattern.empty()) return hits;
    auto lps = buildLps(pattern);
    int n = static_cast<int>(text.size()), m = static_cast<int>(pattern.size());
    int i = 0, q = 0;
    while (i < n) {
        if (text[i] == pattern[q]) {
            ++i;
            ++q;
            if (q == m) {
                hits.push_back(i - m);
                q = lps[q - 1];
            }
        } else if (q > 0) {
            q = lps[q - 1];
        } else {
            ++i;
        }
    }
    return hits;
}

int kmpFirst(const std::string& text, const std::string& pattern) {
    auto hits = kmpSearch(text, pattern);
    return hits.empty() ? -1 : hits[0];
}

int main() {
    auto hits = kmpSearch("ababcabab", "abab");
    if (hits != std::vector<int>{0, 5}) {
        std::cerr << "unexpected hits\n";
        return 1;
    }
    if (kmpFirst("ababcabab", "abab") != 0) {
        std::cerr << "first index failed\n";
        return 1;
    }
    hits = kmpSearch("aaaa", "aa");
    if (hits != std::vector<int>{0, 1, 2}) {
        std::cerr << "overlap hits failed\n";
        return 1;
    }
    if (kmpFirst("hello", "world") != -1) {
        std::cerr << "miss should be -1\n";
        return 1;
    }
    std::cout << "kmp: ok\n";
    return 0;
}
