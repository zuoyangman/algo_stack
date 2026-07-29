fn build_lps(pattern: &str) -> Vec<usize> {
    let p: Vec<char> = pattern.chars().collect();
    let m = p.len();
    let mut lps = vec![0usize; m];
    let mut len = 0usize;
    let mut i = 1usize;
    while i < m {
        if p[i] == p[len] {
            len += 1;
            lps[i] = len;
            i += 1;
        } else if len > 0 {
            len = lps[len - 1];
        } else {
            lps[i] = 0;
            i += 1;
        }
    }
    lps
}

fn kmp_search(text: &str, pattern: &str) -> Vec<usize> {
    let mut hits = Vec::new();
    if pattern.is_empty() {
        return hits;
    }
    let t: Vec<char> = text.chars().collect();
    let p: Vec<char> = pattern.chars().collect();
    let lps = build_lps(pattern);
    let n = t.len();
    let m = p.len();
    let mut i = 0usize;
    let mut q = 0usize;
    while i < n {
        if t[i] == p[q] {
            i += 1;
            q += 1;
            if q == m {
                hits.push(i - m);
                q = lps[q - 1];
            }
        } else if q > 0 {
            q = lps[q - 1];
        } else {
            i += 1;
        }
    }
    hits
}

fn kmp_first(text: &str, pattern: &str) -> isize {
    match kmp_search(text, pattern).first() {
        Some(&i) => i as isize,
        None => -1,
    }
}

fn main() {
    assert_eq!(kmp_search("ababcabab", "abab"), vec![0, 5]);
    assert_eq!(kmp_first("ababcabab", "abab"), 0);
    assert_eq!(kmp_search("aaaa", "aa"), vec![0, 1, 2]);
    assert_eq!(kmp_first("hello", "world"), -1);
    println!("kmp: ok");
}
