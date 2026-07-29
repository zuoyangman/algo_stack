struct Entry {
    key: String,
    value: i32,
    next: Option<Box<Entry>>,
}

struct HashTable {
    buckets: Vec<Option<Box<Entry>>>,
    size: usize,
}

impl HashTable {
    fn new(capacity: usize) -> Self {
        let n = capacity.max(1);
        let mut buckets = Vec::with_capacity(n);
        for _ in 0..n {
            buckets.push(None);
        }
        Self { buckets, size: 0 }
    }

    fn hash(key: &str) -> usize {
        let mut h: u64 = 1469598103934665603;
        for b in key.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(1099511628211);
        }
        h as usize
    }

    fn index(&self, key: &str) -> usize {
        Self::hash(key) % self.buckets.len()
    }

    fn put(&mut self, key: &str, value: i32) {
        let i = self.index(key);
        let mut cur = self.buckets[i].as_mut();
        while let Some(entry) = cur {
            if entry.key == key {
                entry.value = value;
                return;
            }
            cur = entry.next.as_mut();
        }
        let next = self.buckets[i].take();
        self.buckets[i] = Some(Box::new(Entry {
            key: key.to_string(),
            value,
            next,
        }));
        self.size += 1;
    }

    fn get(&self, key: &str) -> Option<i32> {
        let i = self.index(key);
        let mut cur = self.buckets[i].as_ref();
        while let Some(entry) = cur {
            if entry.key == key {
                return Some(entry.value);
            }
            cur = entry.next.as_ref();
        }
        None
    }

    fn remove(&mut self, key: &str) -> bool {
        let i = self.index(key);
        let mut slot = &mut self.buckets[i];
        loop {
            match slot {
                None => return false,
                Some(entry) if entry.key == key => {
                    let next = entry.next.take();
                    *slot = next;
                    self.size -= 1;
                    return true;
                }
                Some(entry) => {
                    slot = &mut entry.next;
                }
            }
        }
    }
}

fn main() {
    let mut ht = HashTable::new(8);
    ht.put("apple", 1);
    ht.put("banana", 2);
    ht.put("cherry", 3);
    ht.put("apple", 10);
    assert_eq!(ht.get("apple"), Some(10));
    assert_eq!(ht.get("banana"), Some(2));
    assert_eq!(ht.get("missing"), None);
    assert!(ht.remove("banana"));
    assert_eq!(ht.get("banana"), None);
    assert!(!ht.remove("banana"));
    assert_eq!(ht.size, 2);
    println!("hash_table: ok");
}
