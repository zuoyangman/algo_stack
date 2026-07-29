#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

// Return index of first x in a, or -1 if absent.
int search(const std::vector<int>& a, int x) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] == x) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

static void expect(const std::vector<int>& a, int x, int want) {
    int got = search(a, x);
    if (got != want) {
        std::cerr << "search got " << got << " want " << want << "\n";
        std::exit(1);
    }
}

int main() {
    expect({}, 1, -1);
    expect({42}, 42, 0);
    expect({42}, 7, -1);
    expect({3, 1, 4, 1, 5}, 1, 1);
    expect({3, 1, 4, 1, 5}, 5, 4);
    expect({3, 1, 4, 1, 5}, 9, -1);
    std::cout << "linear_search: ok\n";
    return 0;
}
