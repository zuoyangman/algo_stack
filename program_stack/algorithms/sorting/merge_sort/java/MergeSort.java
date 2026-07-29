import java.util.Arrays;

public class MergeSort {
    /** Sort a in non-decreasing order (top-down merge sort). Stable. */
    public static void sort(int[] a) {
        if (a == null || a.length < 2) {
            return;
        }
        int[] buf = new int[a.length];
        sortRange(a, 0, a.length, buf);
    }

    private static void sortRange(int[] a, int lo, int hi, int[] buf) {
        if (hi - lo <= 1) {
            return;
        }
        int mid = lo + (hi - lo) / 2;
        sortRange(a, lo, mid, buf);
        sortRange(a, mid, hi, buf);
        merge(a, lo, mid, hi, buf);
    }

    private static void merge(int[] a, int lo, int mid, int hi, int[] buf) {
        int i = lo;
        int j = mid;
        int k = lo;
        while (i < mid && j < hi) {
            if (a[i] <= a[j]) {
                buf[k++] = a[i++];
            } else {
                buf[k++] = a[j++];
            }
        }
        while (i < mid) {
            buf[k++] = a[i++];
        }
        while (j < hi) {
            buf[k++] = a[j++];
        }
        for (int t = lo; t < hi; t++) {
            a[t] = buf[t];
        }
    }

    private static void expect(int[] input, int[] want) {
        int[] a = Arrays.copyOf(input, input.length);
        sort(a);
        if (!Arrays.equals(a, want)) {
            System.err.println("got " + Arrays.toString(a) + " want " + Arrays.toString(want));
            System.exit(1);
        }
    }

    public static void main(String[] args) {
        expect(new int[] {}, new int[] {});
        expect(new int[] {42}, new int[] {42});
        expect(new int[] {1, 2, 3}, new int[] {1, 2, 3});
        expect(new int[] {3, 2, 1}, new int[] {1, 2, 3});
        expect(new int[] {5, 1, 4, 2, 8}, new int[] {1, 2, 4, 5, 8});
        expect(new int[] {3, 1, 2, 1, 3}, new int[] {1, 1, 2, 3, 3});
        System.out.println("merge_sort: ok");
    }
}
