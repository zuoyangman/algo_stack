#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

static void merge(std::vector<int>& a, std::size_t lo, std::size_t mid, std::size_t hi,
                  std::vector<int>& buf) {
    std::size_t i = lo;
    std::size_t j = mid;
    std::size_t k = lo;
    while (i < mid && j < hi) {
        if (a[i] <= a[j]) {
            buf[k++] = a[i++];
        } else {
            buf[k++] = a[j++];
        }
    }
    while (i < mid) {
        buf[k++] = a[i++];
    }
    while (j < hi) {
        buf[k++] = a[j++];
    }
    for (std::size_t t = lo; t < hi; ++t) {
        a[t] = buf[t];
    }
}

static void sort_range(std::vector<int>& a, std::size_t lo, std::size_t hi,
                       std::vector<int>& buf) {
    if (hi - lo <= 1) {
        return;
    }
    std::size_t mid = lo + (hi - lo) / 2;
    sort_range(a, lo, mid, buf);
    sort_range(a, mid, hi, buf);
    merge(a, lo, mid, hi, buf);
}

// Top-down merge sort (non-decreasing). Stable.
void sort(std::vector<int>& a) {
    if (a.size() < 2) {
        return;
    }
    std::vector<int> buf(a.size());
    sort_range(a, 0, a.size(), buf);
}

static void expect(std::vector<int> input, const std::vector<int>& want) {
    sort(input);
    if (input != want) {
        std::cerr << "mismatch\n";
        std::exit(1);
    }
}

int main() {
    expect({}, {});
    expect({42}, {42});
    expect({1, 2, 3}, {1, 2, 3});
    expect({3, 2, 1}, {1, 2, 3});
    expect({5, 1, 4, 2, 8}, {1, 2, 4, 5, 8});
    expect({3, 1, 2, 1, 3}, {1, 1, 2, 3, 3});
    std::cout << "merge_sort: ok\n";
    return 0;
}
