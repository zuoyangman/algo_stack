package main

import (
	"fmt"
	"os"
)

type node struct {
	value int
	next  *node
}

type Queue struct {
	head *node
	tail *node
}

func (q *Queue) Enqueue(x int) {
	n := &node{value: x}
	if q.tail == nil {
		q.head = n
		q.tail = n
	} else {
		q.tail.next = n
		q.tail = n
	}
}

func (q *Queue) Dequeue() (int, bool) {
	if q.head == nil {
		return 0, false
	}
	v := q.head.value
	q.head = q.head.next
	if q.head == nil {
		q.tail = nil
	}
	return v, true
}

func (q *Queue) Front() (int, bool) {
	if q.head == nil {
		return 0, false
	}
	return q.head.value, true
}

func (q *Queue) IsEmpty() bool {
	return q.head == nil
}

func expect(cond bool, msg string) {
	if !cond {
		fmt.Fprintln(os.Stderr, "FAIL:", msg)
		os.Exit(1)
	}
}

func main() {
	q := &Queue{}
	expect(q.IsEmpty(), "empty initially")
	q.Enqueue(1)
	q.Enqueue(2)
	q.Enqueue(3)
	f, ok := q.Front()
	expect(ok && f == 1, "front")
	v, ok := q.Dequeue()
	expect(ok && v == 1, "dequeue 1")
	v, ok = q.Dequeue()
	expect(ok && v == 2, "dequeue 2")
	q.Enqueue(4)
	f, ok = q.Front()
	expect(ok && f == 3, "front 3")
	v, ok = q.Dequeue()
	expect(ok && v == 3, "dequeue 3")
	v, ok = q.Dequeue()
	expect(ok && v == 4, "dequeue 4")
	expect(q.IsEmpty(), "empty finally")
	fmt.Println("queue: ok")
}
