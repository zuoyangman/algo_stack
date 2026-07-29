#include <cstdlib>
#include <iostream>
#include <stdexcept>

struct Node {
    int value;
    Node* next;
    explicit Node(int v) : value(v), next(nullptr) {}
};

class Queue {
    Node* head_ = nullptr;
    Node* tail_ = nullptr;

public:
    Queue() = default;

    ~Queue() {
        while (head_) {
            Node* n = head_;
            head_ = head_->next;
            delete n;
        }
    }

    Queue(const Queue&) = delete;
    Queue& operator=(const Queue&) = delete;

    void enqueue(int x) {
        Node* n = new Node(x);
        if (!tail_) {
            head_ = tail_ = n;
        } else {
            tail_->next = n;
            tail_ = n;
        }
    }

    int dequeue() {
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
        return v;
    }

    int front() const {
        if (!head_) {
            throw std::runtime_error("empty");
        }
        return head_->value;
    }

    bool is_empty() const { return head_ == nullptr; }
};

static void expect(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        std::exit(1);
    }
}

int main() {
    Queue q;
    expect(q.is_empty(), "empty initially");
    q.enqueue(1);
    q.enqueue(2);
    q.enqueue(3);
    expect(q.front() == 1, "front");
    expect(q.dequeue() == 1, "dequeue 1");
    expect(q.dequeue() == 2, "dequeue 2");
    q.enqueue(4);
    expect(q.front() == 3, "front 3");
    expect(q.dequeue() == 3, "dequeue 3");
    expect(q.dequeue() == 4, "dequeue 4");
    expect(q.is_empty(), "empty finally");
    std::cout << "queue: ok\n";
    return 0;
}
