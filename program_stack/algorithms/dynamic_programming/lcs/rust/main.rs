fn lcs_length(a: &str, b: &str) -> usize {
    let aa: Vec<char> = a.chars().collect();
    let bb: Vec<char> = b.chars().collect();
    let n = aa.len();
    let m = bb.len();
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in 1..=n {
        for j in 1..=m {
            if aa[i - 1] == bb[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }
    dp[n][m]
}

fn lcs_string(a: &str, b: &str) -> String {
    let aa: Vec<char> = a.chars().collect();
    let bb: Vec<char> = b.chars().collect();
    let n = aa.len();
    let m = bb.len();
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in 1..=n {
        for j in 1..=m {
            if aa[i - 1] == bb[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }
    let mut out = Vec::new();
    let mut i = n;
    let mut j = m;
    while i > 0 && j > 0 {
        if aa[i - 1] == bb[j - 1] {
            out.push(aa[i - 1]);
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] >= dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    out.reverse();
    out.into_iter().collect()
}

fn main() {
    let a = "ABCBDAB";
    let b = "BDCABA";
    assert_eq!(lcs_length(a, b), 4);
    let s = lcs_string(a, b);
    assert_eq!(s.len(), 4);
    assert_eq!(lcs_length(&s, a), 4);
    assert_eq!(lcs_length(&s, b), 4);
    assert_eq!(lcs_length("", "xyz"), 0);
    assert_eq!(lcs_length("abc", "abc"), 3);
    println!("lcs: ok");
}
