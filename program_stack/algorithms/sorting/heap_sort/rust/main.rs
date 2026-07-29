fn sift_down(a: &mut [i32], heap_size: usize, mut i: usize) {
    loop {
        let mut largest = i;
        let left = 2 * i + 1;
        let right = 2 * i + 2;
        if left < heap_size && a[left] > a[largest] {
            largest = left;
        }
        if right < heap_size && a[right] > a[largest] {
            largest = right;
        }
        if largest == i {
            return;
        }
        a.swap(i, largest);
        i = largest;
    }
}

fn sort(a: &mut [i32]) {
    if a.len() < 2 {
        return;
    }
    let n = a.len();
    for i in (0..n / 2).rev() {
        sift_down(a, n, i);
    }
    for end in (1..n).rev() {
        a.swap(0, end);
        sift_down(a, end, 0);
    }
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
    println!("heap_sort: ok");
}
