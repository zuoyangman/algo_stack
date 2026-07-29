#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

// In-place selection sort (non-decreasing). Not stable.
void sort(std::vector<int>& a) {
    if (a.size() < 2) {
        return;
    }
    for (std::size_t i = 0; i + 1 < a.size(); ++i) {
        std::size_t min = i;
        for (std::size_t j = i + 1; j < a.size(); ++j) {
            if (a[j] < a[min]) {
                min = j;
            }
        }
        if (min != i) {
            std::swap(a[i], a[min]);
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
    std::cout << "selection_sort: ok\n";
    return 0;
}
