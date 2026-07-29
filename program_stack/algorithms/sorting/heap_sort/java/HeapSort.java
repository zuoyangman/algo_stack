import java.util.Arrays;

public class HeapSort {
    /** In-place heapsort (non-decreasing). Not stable. */
    public static void sort(int[] a) {
        if (a == null || a.length < 2) {
            return;
        }
        int n = a.length;
        for (int i = n / 2 - 1; i >= 0; i--) {
            siftDown(a, n, i);
        }
        for (int end = n - 1; end > 0; end--) {
            int tmp = a[0];
            a[0] = a[end];
            a[end] = tmp;
            siftDown(a, end, 0);
        }
    }

    private static void siftDown(int[] a, int heapSize, int i) {
        while (true) {
            int largest = i;
            int left = 2 * i + 1;
            int right = 2 * i + 2;
            if (left < heapSize && a[left] > a[largest]) {
                largest = left;
            }
            if (right < heapSize && a[right] > a[largest]) {
                largest = right;
            }
            if (largest == i) {
                return;
            }
            int tmp = a[i];
            a[i] = a[largest];
            a[largest] = tmp;
            i = largest;
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
        System.out.println("heap_sort: ok");
    }
}
