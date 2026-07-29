#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

static int partition(std::vector<int>& a, int lo, int hi) {
    int pivot = a[hi];
    int i = lo;
    for (int j = lo; j < hi; ++j) {
        if (a[j] <= pivot) {
            std::swap(a[i], a[j]);
            ++i;
        }
    }
    std::swap(a[i], a[hi]);
    return i;
}

static void sort_range(std::vector<int>& a, int lo, int hi) {
    if (lo >= hi) {
        return;
    }
    int p = partition(a, lo, hi);
    sort_range(a, lo, p - 1);
    sort_range(a, p + 1, hi);
}

// In-place quicksort (Lomuto, last-element pivot). Not stable.
void sort(std::vector<int>& a) {
    if (a.size() < 2) {
        return;
    }
    sort_range(a, 0, static_cast<int>(a.size()) - 1);
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
    std::cout << "quick_sort: ok\n";
    return 0;
}
