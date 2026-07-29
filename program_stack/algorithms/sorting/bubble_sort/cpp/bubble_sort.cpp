#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

// In-place bubble sort (non-decreasing). Stable.
void sort(std::vector<int>& a) {
    if (a.size() < 2) {
        return;
    }
    for (std::size_t end = a.size() - 1; end > 0; --end) {
        bool swapped = false;
        for (std::size_t i = 0; i < end; ++i) {
            if (a[i] > a[i + 1]) {
                std::swap(a[i], a[i + 1]);
                swapped = true;
            }
        }
        if (!swapped) {
            break;
        }
    }
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
    std::cout << "bubble_sort: ok\n";
    return 0;
}
