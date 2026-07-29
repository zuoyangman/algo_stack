import java.util.Arrays;

public class QuickSort {
    /** In-place quicksort (Lomuto partition, last-element pivot). Not stable. */
    public static void sort(int[] a) {
        if (a == null || a.length < 2) {
            return;
        }
        sortRange(a, 0, a.length - 1);
    }

    private static void sortRange(int[] a, int lo, int hi) {
        if (lo >= hi) {
            return;
        }
        int p = partition(a, lo, hi);
        sortRange(a, lo, p - 1);
        sortRange(a, p + 1, hi);
    }

    private static int partition(int[] a, int lo, int hi) {
        int pivot = a[hi];
        int i = lo;
        for (int j = lo; j < hi; j++) {
            if (a[j] <= pivot) {
                int tmp = a[i];
                a[i] = a[j];
                a[j] = tmp;
                i++;
            }
        }
        int tmp = a[i];
        a[i] = a[hi];
        a[hi] = tmp;
        return i;
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
        System.out.println("quick_sort: ok");
    }
}
