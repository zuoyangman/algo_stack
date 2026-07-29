fn merge(a: &mut [i32], lo: usize, mid: usize, hi: usize, buf: &mut [i32]) {
    let mut i = lo;
    let mut j = mid;
    let mut k = lo;
    while i < mid && j < hi {
        if a[i] <= a[j] {
            buf[k] = a[i];
            i += 1;
        } else {
            buf[k] = a[j];
            j += 1;
        }
        k += 1;
    }
    while i < mid {
        buf[k] = a[i];
        i += 1;
        k += 1;
    }
    while j < hi {
        buf[k] = a[j];
        j += 1;
        k += 1;
    }
    for t in lo..hi {
        a[t] = buf[t];
    }
}

fn sort_range(a: &mut [i32], lo: usize, hi: usize, buf: &mut [i32]) {
    if hi - lo <= 1 {
        return;
    }
    let mid = lo + (hi - lo) / 2;
    sort_range(a, lo, mid, buf);
    sort_range(a, mid, hi, buf);
    merge(a, lo, mid, hi, buf);
}

fn sort(a: &mut [i32]) {
    if a.len() < 2 {
        return;
    }
    let mut buf = vec![0; a.len()];
    let n = a.len();
    sort_range(a, 0, n, &mut buf);
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
    println!("merge_sort: ok");
}
