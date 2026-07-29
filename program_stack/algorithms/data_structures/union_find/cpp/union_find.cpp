#include <cstdlib>
#include <iostream>
#include <vector>

class UnionFind {
    std::vector<int> parent_;
    std::vector<int> rank_;

public:
    explicit UnionFind(int n) : parent_(n), rank_(n, 0) {
        for (int i = 0; i < n; ++i) {
            parent_[i] = i;
        }
    }

    int find(int x) {
        if (parent_[x] != x) {
            parent_[x] = find(parent_[x]);
        }
        return parent_[x];
    }

    bool unite(int a, int b) {
        int ra = find(a);
        int rb = find(b);
        if (ra == rb) {
            return false;
        }
        if (rank_[ra] < rank_[rb]) {
            parent_[ra] = rb;
        } else if (rank_[ra] > rank_[rb]) {
            parent_[rb] = ra;
        } else {
            parent_[rb] = ra;
            ++rank_[ra];
        }
        return true;
    }

    bool connected(int a, int b) { return find(a) == find(b); }
};

static void expect(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        std::exit(1);
    }
}

int main() {
    UnionFind uf(6);
    expect(!uf.connected(0, 1), "initially separate");
    expect(uf.unite(0, 1), "union 0-1");
    expect(uf.unite(1, 2), "union 1-2");
    expect(uf.connected(0, 2), "0 connected 2");
    expect(!uf.connected(0, 3), "0 not 3");
    expect(uf.unite(3, 4), "union 3-4");
    expect(uf.unite(2, 4), "merge components");
    expect(uf.connected(0, 3), "0 connected 3");
    expect(!uf.unite(0, 2), "already united");
    expect(uf.find(5) == 5, "singleton 5");
    std::cout << "union_find: ok\n";
    return 0;
}
