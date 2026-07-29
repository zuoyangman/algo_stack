#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

struct Node {
    std::unordered_map<char, std::unique_ptr<Node>> next;
    bool end = false;
};

class Trie {
public:
    void insert(const std::string& word) {
        Node* cur = &root;
        for (char c : word) {
            auto& slot = cur->next[c];
            if (!slot) slot = std::make_unique<Node>();
            cur = slot.get();
        }
        cur->end = true;
    }

    bool search(const std::string& word) const {
        const Node* cur = walk(word);
        return cur && cur->end;
    }

    bool startsWith(const std::string& prefix) const {
        return walk(prefix) != nullptr;
    }

private:
    Node root;

    const Node* walk(const std::string& s) const {
        const Node* cur = &root;
        for (char c : s) {
            auto it = cur->next.find(c);
            if (it == cur->next.end()) return nullptr;
            cur = it->second.get();
        }
        return cur;
    }
};

int main() {
    Trie t;
    t.insert("apple");
    if (!t.search("apple") || t.search("app") || !t.startsWith("app")) {
        std::cerr << "apple/app checks failed\n";
        return 1;
    }
    t.insert("app");
    if (!t.search("app") || !t.startsWith("ap") || t.search("appl")) {
        std::cerr << "after insert app failed\n";
        return 1;
    }
    if (t.search("") || t.startsWith("b")) {
        std::cerr << "edge checks failed\n";
        return 1;
    }
    std::cout << "trie: ok\n";
    return 0;
}
