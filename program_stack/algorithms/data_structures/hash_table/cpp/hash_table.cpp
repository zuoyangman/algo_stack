#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

struct Entry {
    std::string key;
    int value;
    Entry* next;
    Entry(std::string k, int v, Entry* n) : key(std::move(k)), value(v), next(n) {}
};

class HashTable {
    std::vector<Entry*> buckets_;
    int size_ = 0;

    static std::size_t hash_str(const std::string& s) {
        std::size_t h = 1469598103934665603ull;
        for (unsigned char c : s) {
            h ^= c;
            h *= 1099511628211ull;
        }
        return h;
    }

    std::size_t index(const std::string& key) const {
        return hash_str(key) % buckets_.size();
    }

public:
    explicit HashTable(std::size_t capacity) : buckets_(capacity ? capacity : 1, nullptr) {}

    HashTable() : HashTable(16) {}

    ~HashTable() {
        for (Entry* e : buckets_) {
            while (e) {
                Entry* n = e->next;
                delete e;
                e = n;
            }
        }
    }

    HashTable(const HashTable&) = delete;
    HashTable& operator=(const HashTable&) = delete;

    void put(const std::string& key, int value) {
        std::size_t i = index(key);
        for (Entry* e = buckets_[i]; e; e = e->next) {
            if (e->key == key) {
                e->value = value;
                return;
            }
        }
        buckets_[i] = new Entry(key, value, buckets_[i]);
        ++size_;
    }

    std::optional<int> get(const std::string& key) const {
        std::size_t i = index(key);
        for (Entry* e = buckets_[i]; e; e = e->next) {
            if (e->key == key) {
                return e->value;
            }
        }
        return std::nullopt;
    }

    bool remove(const std::string& key) {
        std::size_t i = index(key);
        Entry* prev = nullptr;
        Entry* cur = buckets_[i];
        while (cur) {
            if (cur->key == key) {
                if (!prev) {
                    buckets_[i] = cur->next;
                } else {
                    prev->next = cur->next;
                }
                delete cur;
                --size_;
                return true;
            }
            prev = cur;
            cur = cur->next;
        }
        return false;
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
    HashTable ht(8);
    ht.put("apple", 1);
    ht.put("banana", 2);
    ht.put("cherry", 3);
    ht.put("apple", 10);
    expect(ht.get("apple") == 10, "get apple");
    expect(ht.get("banana") == 2, "get banana");
    expect(!ht.get("missing").has_value(), "missing");
    expect(ht.remove("banana"), "remove banana");
    expect(!ht.get("banana").has_value(), "banana gone");
    expect(!ht.remove("banana"), "remove again");
    expect(ht.size() == 2, "size");
    std::cout << "hash_table: ok\n";
    return 0;
}
