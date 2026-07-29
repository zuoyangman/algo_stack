#include <cstdlib>
#include <iostream>
#include <vector>

struct Node {
    int key;
    Node* left = nullptr;
    Node* right = nullptr;
    explicit Node(int k) : key(k) {}
};

class BST {
    Node* root_ = nullptr;

    static Node* insert_rec(Node* n, int x) {
        if (!n) {
            return new Node(x);
        }
        if (x < n->key) {
            n->left = insert_rec(n->left, x);
        } else if (x > n->key) {
            n->right = insert_rec(n->right, x);
        }
        return n;
    }

    static void destroy(Node* n) {
        if (!n) {
            return;
        }
        destroy(n->left);
        destroy(n->right);
        delete n;
    }

    static void inorder_rec(Node* n, std::vector<int>& out) {
        if (!n) {
            return;
        }
        inorder_rec(n->left, out);
        out.push_back(n->key);
        inorder_rec(n->right, out);
    }

public:
    BST() = default;
    ~BST() { destroy(root_); }
    BST(const BST&) = delete;
    BST& operator=(const BST&) = delete;

    void insert(int x) { root_ = insert_rec(root_, x); }

    bool contains(int x) const {
        Node* cur = root_;
        while (cur) {
            if (x == cur->key) {
                return true;
            }
            cur = x < cur->key ? cur->left : cur->right;
        }
        return false;
    }

    std::vector<int> inorder() const {
        std::vector<int> out;
        inorder_rec(root_, out);
        return out;
    }
};

static void expect(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        std::exit(1);
    }
}

int main() {
    BST t;
    for (int v : {5, 3, 7, 1, 4, 6, 8}) {
        t.insert(v);
    }
    t.insert(5);
    expect(t.contains(4), "contains 4");
    expect(!t.contains(2), "missing 2");
    expect(t.inorder() == std::vector<int>({1, 3, 4, 5, 6, 7, 8}), "inorder");
    std::cout << "binary_search_tree: ok\n";
    return 0;
}
