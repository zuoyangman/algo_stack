public class HashTableDemo {
    static final class Entry {
        final String key;
        int value;
        Entry next;

        Entry(String key, int value, Entry next) {
            this.key = key;
            this.value = value;
            this.next = next;
        }
    }

    static final class HashTable {
        private final Entry[] buckets;
        private int size;

        HashTable(int capacity) {
            buckets = new Entry[Math.max(1, capacity)];
        }

        private int index(String key) {
            int h = key.hashCode();
            return (h & 0x7fffffff) % buckets.length;
        }

        void put(String key, int value) {
            int i = index(key);
            for (Entry e = buckets[i]; e != null; e = e.next) {
                if (e.key.equals(key)) {
                    e.value = value;
                    return;
                }
            }
            buckets[i] = new Entry(key, value, buckets[i]);
            size++;
        }

        Integer get(String key) {
            int i = index(key);
            for (Entry e = buckets[i]; e != null; e = e.next) {
                if (e.key.equals(key)) {
                    return e.value;
                }
            }
            return null;
        }

        boolean remove(String key) {
            int i = index(key);
            Entry prev = null;
            Entry cur = buckets[i];
            while (cur != null) {
                if (cur.key.equals(key)) {
                    if (prev == null) {
                        buckets[i] = cur.next;
                    } else {
                        prev.next = cur.next;
                    }
                    size--;
                    return true;
                }
                prev = cur;
                cur = cur.next;
            }
            return false;
        }

        int size() {
            return size;
        }
    }

    static void expect(boolean cond, String msg) {
        if (!cond) {
            System.err.println("FAIL: " + msg);
            System.exit(1);
        }
    }

    public static void main(String[] args) {
        HashTable ht = new HashTable(8);
        ht.put("apple", 1);
        ht.put("banana", 2);
        ht.put("cherry", 3);
        ht.put("apple", 10); // update
        expect(ht.get("apple") == 10, "get apple");
        expect(ht.get("banana") == 2, "get banana");
        expect(ht.get("missing") == null, "missing");
        expect(ht.remove("banana"), "remove banana");
        expect(ht.get("banana") == null, "banana gone");
        expect(!ht.remove("banana"), "remove again");
        expect(ht.size() == 2, "size");
        System.out.println("hash_table: ok");
    }
}
