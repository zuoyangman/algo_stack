import java.util.Arrays;

public class InsertionSort {
    /** In-place insertion sort (non-decreasing). Stable. */
    public static void sort(int[] a) {
        if (a == null || a.length < 2) {
            return;
        }
        for (int i = 1; i < a.length; i++) {
            int key = a[i];
            int j = i - 1;
            while (j >= 0 && a[j] > key) {
                a[j + 1] = a[j];
                j--;
            }
            a[j + 1] = key;
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
        System.out.println("insertion_sort: ok");
    }
}
