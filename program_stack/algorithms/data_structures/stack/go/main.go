package main

import (
	"fmt"
	"os"
)

type Stack struct {
	data []rune
}

func (s *Stack) Push(x rune) {
	s.data = append(s.data, x)
}

func (s *Stack) Pop() (rune, bool) {
	if s.IsEmpty() {
		return 0, false
	}
	i := len(s.data) - 1
	v := s.data[i]
	s.data = s.data[:i]
	return v, true
}

func (s *Stack) Peek() (rune, bool) {
	if s.IsEmpty() {
		return 0, false
	}
	return s.data[len(s.data)-1], true
}

func (s *Stack) IsEmpty() bool {
	return len(s.data) == 0
}

func isBalanced(str string) bool {
	st := &Stack{}
	for _, c := range str {
		switch c {
		case '(', '[', '{':
			st.Push(c)
		case ')', ']', '}':
			open, ok := st.Pop()
			if !ok {
				return false
			}
			if (c == ')' && open != '(') || (c == ']' && open != '[') || (c == '}' && open != '{') {
				return false
			}
		}
	}
	return st.IsEmpty()
}

func reverse(str string) string {
	st := &Stack{}
	for _, c := range str {
		st.Push(c)
	}
	out := make([]rune, 0, len(str))
	for !st.IsEmpty() {
		v, _ := st.Pop()
		out = append(out, v)
	}
	return string(out)
}

func expect(cond bool, msg string) {
	if !cond {
		fmt.Fprintln(os.Stderr, "FAIL:", msg)
		os.Exit(1)
	}
}

func main() {
	st := &Stack{}
	expect(st.IsEmpty(), "empty initially")
	st.Push('a')
	st.Push('b')
	p, ok := st.Peek()
	expect(ok && p == 'b', "peek")
	v, ok := st.Pop()
	expect(ok && v == 'b', "pop")
	v, ok = st.Pop()
	expect(ok && v == 'a', "pop a")
	expect(st.IsEmpty(), "empty after pops")
	expect(isBalanced("({[]})"), "balanced")
	expect(!isBalanced("([)]"), "unbalanced")
	expect(reverse("abc") == "cba", "reverse")
	fmt.Println("stack: ok")
}
