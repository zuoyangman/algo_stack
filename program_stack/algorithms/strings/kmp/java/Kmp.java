import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Kmp {
    /** LPS / failure function for pattern. */
    public static int[] buildLps(String pattern) {
        int m = pattern.length();
        int[] lps = new int[m];
        int len = 0, i = 1;
        while (i < m) {
            if (pattern.charAt(i) == pattern.charAt(len)) {
                lps[i++] = ++len;
            } else if (len > 0) {
                len = lps[len - 1];
            } else {
                lps[i++] = 0;
            }
        }
        return lps;
    }

    /** All starting indices of pattern in text. */
    public static List<Integer> kmpSearch(String text, String pattern) {
        List<Integer> hits = new ArrayList<>();
        if (pattern.isEmpty()) return hits;
        int[] lps = buildLps(pattern);
        int n = text.length(), m = pattern.length();
        int i = 0, q = 0;
        while (i < n) {
            if (text.charAt(i) == pattern.charAt(q)) {
                i++;
                q++;
                if (q == m) {
                    hits.add(i - m);
                    q = lps[q - 1];
                }
            } else if (q > 0) {
                q = lps[q - 1];
            } else {
                i++;
            }
        }
        return hits;
    }

    /** First match index, or -1. */
    public static int kmpFirst(String text, String pattern) {
        List<Integer> hits = kmpSearch(text, pattern);
        return hits.isEmpty() ? -1 : hits.get(0);
    }

    public static void main(String[] args) {
        List<Integer> hits = kmpSearch("ababcabab", "abab");
        if (!hits.equals(Arrays.asList(0, 5))) {
            System.err.println("unexpected hits: " + hits);
            System.exit(1);
        }
        if (kmpFirst("ababcabab", "abab") != 0) {
            System.err.println("first index failed");
            System.exit(1);
        }
        hits = kmpSearch("aaaa", "aa");
        if (!hits.equals(Arrays.asList(0, 1, 2))) {
            System.err.println("overlap hits: " + hits);
            System.exit(1);
        }
        if (kmpFirst("hello", "world") != -1) {
            System.err.println("miss should be -1");
            System.exit(1);
        }
        System.out.println("kmp: ok");
    }
}
