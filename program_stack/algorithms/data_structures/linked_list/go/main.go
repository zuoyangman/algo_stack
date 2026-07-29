package main

import (
	"fmt"
	"os"
)

type node struct {
	value int
	next  *node
}

type LinkedList struct {
	head *node
	tail *node
	size int
}

func (l *LinkedList) PushFront(x int) {
	n := &node{value: x, next: l.head}
	l.head = n
	if l.tail == nil {
		l.tail = n
	}
	l.size++
}

func (l *LinkedList) PushBack(x int) {
	n := &node{value: x}
	if l.tail == nil {
		l.head = n
		l.tail = n
	} else {
		l.tail.next = n
		l.tail = n
	}
	l.size++
}

func (l *LinkedList) PopFront() (int, bool) {
	if l.head == nil {
		return 0, false
	}
	v := l.head.value
	l.head = l.head.next
	if l.head == nil {
		l.tail = nil
	}
	l.size--
	return v, true
}

func (l *LinkedList) Find(x int) bool {
	for cur := l.head; cur != nil; cur = cur.next {
		if cur.value == x {
			return true
		}
	}
	return false
}

func (l *LinkedList) ToArray() []int {
	out := make([]int, 0, l.size)
	for cur := l.head; cur != nil; cur = cur.next {
		out = append(out, cur.value)
	}
	return out
}

func expect(cond bool, msg string) {
	if !cond {
		fmt.Fprintln(os.Stderr, "FAIL:", msg)
		os.Exit(1)
	}
}

func equal(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func main() {
	list := &LinkedList{}
	list.PushBack(2)
	list.PushFront(1)
	list.PushBack(3)
	expect(equal(list.ToArray(), []int{1, 2, 3}), "values after pushes")
	expect(list.Find(2), "find 2")
	expect(!list.Find(9), "missing 9")
	v, ok := list.PopFront()
	expect(ok && v == 1, "pop_front")
	expect(equal(list.ToArray(), []int{2, 3}), "after pop")
	expect(list.size == 2, "size")
	fmt.Println("linked_list: ok")
}
