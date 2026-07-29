package main

import (
	"fmt"
	"os"
)

type entry struct {
	key   string
	value int
	next  *entry
}

type HashTable struct {
	buckets []*entry
	size    int
}

func NewHashTable(capacity int) *HashTable {
	if capacity < 1 {
		capacity = 1
	}
	return &HashTable{buckets: make([]*entry, capacity)}
}

func hashStr(s string) uint64 {
	var h uint64 = 1469598103934665603
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= 1099511628211
	}
	return h
}

func (ht *HashTable) index(key string) int {
	return int(hashStr(key) % uint64(len(ht.buckets)))
}

func (ht *HashTable) Put(key string, value int) {
	i := ht.index(key)
	for e := ht.buckets[i]; e != nil; e = e.next {
		if e.key == key {
			e.value = value
			return
		}
	}
	ht.buckets[i] = &entry{key: key, value: value, next: ht.buckets[i]}
	ht.size++
}

func (ht *HashTable) Get(key string) (int, bool) {
	i := ht.index(key)
	for e := ht.buckets[i]; e != nil; e = e.next {
		if e.key == key {
			return e.value, true
		}
	}
	return 0, false
}

func (ht *HashTable) Remove(key string) bool {
	i := ht.index(key)
	var prev *entry
	cur := ht.buckets[i]
	for cur != nil {
		if cur.key == key {
			if prev == nil {
				ht.buckets[i] = cur.next
			} else {
				prev.next = cur.next
			}
			ht.size--
			return true
		}
		prev = cur
		cur = cur.next
	}
	return false
}

func expect(cond bool, msg string) {
	if !cond {
		fmt.Fprintln(os.Stderr, "FAIL:", msg)
		os.Exit(1)
	}
}

func main() {
	ht := NewHashTable(8)
	ht.Put("apple", 1)
	ht.Put("banana", 2)
	ht.Put("cherry", 3)
	ht.Put("apple", 10)
	v, ok := ht.Get("apple")
	expect(ok && v == 10, "get apple")
	v, ok = ht.Get("banana")
	expect(ok && v == 2, "get banana")
	_, ok = ht.Get("missing")
	expect(!ok, "missing")
	expect(ht.Remove("banana"), "remove banana")
	_, ok = ht.Get("banana")
	expect(!ok, "banana gone")
	expect(!ht.Remove("banana"), "remove again")
	expect(ht.size == 2, "size")
	fmt.Println("hash_table: ok")
}
