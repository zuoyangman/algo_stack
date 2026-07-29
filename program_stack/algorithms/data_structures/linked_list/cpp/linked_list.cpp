#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

struct Node {
    int value;
    Node* next;
    explicit Node(int v) : value(v), next(nullptr) {}
};

class LinkedList {
    Node* head_ = nullptr;
    Node* tail_ = nullptr;
    int size_ = 0;

public:
    LinkedList() = default;

    ~LinkedList() {
        while (head_) {
            Node* n = head_;
            head_ = head_->next;
            delete n;
        }
    }

    LinkedList(const LinkedList&) = delete;
    LinkedList& operator=(const LinkedList&) = delete;

    void push_front(int x) {
        Node* n = new Node(x);
        n->next = head_;
        head_ = n;
        if (!tail_) {
            tail_ = n;
        }
        ++size_;
    }

    void push_back(int x) {
        Node* n = new Node(x);
        if (!tail_) {
            head_ = tail_ = n;
        } else {
            tail_->next = n;
            tail_ = n;
        }
        ++size_;
    }

    int pop_front() {
        if (!head_) {
            throw std::runtime_error("empty");
        }
        Node* n = head_;
        int v = n->value;
        head_ = head_->next;
        delete n;
        if (!head_) {
            tail_ = nullptr;
        }
        --size_;
        return v;
    }

    bool find(int x) const {
        for (Node* cur = head_; cur; cur = cur->next) {
            if (cur->value == x) {
                return true;
            }
        }
        return false;
    }

    std::vector<int> to_array() const {
        std::vector<int> out;
        out.reserve(size_);
        for (Node* cur = head_; cur; cur = cur->next) {
            out.push_back(cur->value);
        }
        return out;
    }

    int size() const { return size_; }
};

static void expect(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        std::exit(1);
    }
}

int main() {
    LinkedList list;
    list.push_back(2);
    list.push_front(1);
    list.push_back(3);
    expect(list.to_array() == std::vector<int>({1, 2, 3}), "values after pushes");
    expect(list.find(2), "find 2");
    expect(!list.find(9), "missing 9");
    expect(list.pop_front() == 1, "pop_front");
    expect(list.to_array() == std::vector<int>({2, 3}), "after pop");
    expect(list.size() == 2, "size");
    std::cout << "linked_list: ok\n";
    return 0;
}
