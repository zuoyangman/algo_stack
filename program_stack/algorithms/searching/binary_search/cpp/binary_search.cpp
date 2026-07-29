#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

// On a sorted (non-decreasing) array, return an index of x, or -1 if absent.
int search(const std::vector<int>& a, int x) {
    if (a.empty()) {
        return -1;
    }
    int lo = 0;
    int hi = static_cast<int>(a.size()) - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (a[mid] == x) {
            return mid;
        } else if (a[mid] < x) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return -1;
}

static void expect_exact(const std::vector<int>& a, int x, int want) {
    int got = search(a, x);
    if (got != want) {
        std::cerr << "search got " << got << " want " << want << "\n";
        std::exit(1);
    }
}

static void expect_hit(const std::vector<int>& a, int x) {
    int got = search(a, x);
    if (got < 0 || static_cast<std::size_t>(got) >= a.size() || a[got] != x) {
        std::cerr << "expected hit for " << x << "\n";
        std::exit(1);
    }
}

int main() {
    expect_exact({}, 1, -1);
    expect_exact({42}, 42, 0);
    expect_exact({42}, 7, -1);
    expect_hit({1, 2, 3, 4, 5}, 1);
    expect_hit({1, 2, 3, 4, 5}, 5);
    expect_hit({1, 2, 3, 4, 5}, 3);
    expect_exact({1, 2, 3, 4, 5}, 6, -1);
    expect_hit({1, 1, 2, 2, 3}, 2);
    std::cout << "binary_search: ok\n";
    return 0;
}
