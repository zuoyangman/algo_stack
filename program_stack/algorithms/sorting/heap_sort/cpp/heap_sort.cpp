#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

static void sift_down(std::vector<int>& a, std::size_t heap_size, std::size_t i) {
    while (true) {
        std::size_t largest = i;
        std::size_t left = 2 * i + 1;
        std::size_t right = 2 * i + 2;
        if (left < heap_size && a[left] > a[largest]) {
            largest = left;
        }
        if (right < heap_size && a[right] > a[largest]) {
            largest = right;
        }
        if (largest == i) {
            return;
        }
        std::swap(a[i], a[largest]);
        i = largest;
    }
}

// In-place heapsort (non-decreasing). Not stable.
void sort(std::vector<int>& a) {
    if (a.size() < 2) {
        return;
    }
    std::size_t n = a.size();
    for (std::size_t i = n / 2; i-- > 0;) {
        sift_down(a, n, i);
    }
    for (std::size_t end = n - 1; end > 0; --end) {
        std::swap(a[0], a[end]);
        sift_down(a, end, 0);
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
    std::cout << "heap_sort: ok\n";
    return 0;
}
