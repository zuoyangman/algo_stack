fn partition(a: &mut [i32], lo: usize, hi: usize) -> usize {
    let pivot = a[hi];
    let mut i = lo;
    for j in lo..hi {
        if a[j] <= pivot {
            a.swap(i, j);
            i += 1;
        }
    }
    a.swap(i, hi);
    i
}

fn sort_range(a: &mut [i32], lo: isize, hi: isize) {
    if lo >= hi {
        return;
    }
    let p = partition(a, lo as usize, hi as usize) as isize;
    sort_range(a, lo, p - 1);
    sort_range(a, p + 1, hi);
}

fn sort(a: &mut [i32]) {
    if a.len() < 2 {
        return;
    }
    sort_range(a, 0, (a.len() - 1) as isize);
}

fn expect(input: &[i32], want: &[i32]) {
    let mut a = input.to_vec();
    sort(&mut a);
    assert_eq!(a, want);
}

fn main() {
    expect(&[], &[]);
    expect(&[42], &[42]);
    expect(&[1, 2, 3], &[1, 2, 3]);
    expect(&[3, 2, 1], &[1, 2, 3]);
    expect(&[5, 1, 4, 2, 8], &[1, 2, 4, 5, 8]);
    expect(&[3, 1, 2, 1, 3], &[1, 1, 2, 3, 3]);
    println!("quick_sort: ok");
}
