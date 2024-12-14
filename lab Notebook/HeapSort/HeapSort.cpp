#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Function to heapify a subtree rooted with node i
void heapify(vector<int>& arr, int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && arr[left] > arr[largest])
        largest = left;

    if (right < n && arr[right] > arr[largest])
        largest = right;

    if (largest != i) {
        swap(arr[i], arr[largest]);
        heapify(arr, n, largest);
    }
}

// Function to perform heap sort
void heapSort(vector<int>& arr) {
    int n = arr.size();

    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);

    for (int i = n - 1; i > 0; i--) {
        swap(arr[0], arr[i]);
        heapify(arr, i, 0);
    }
}

// Function to print an array
void printArray(const vector<int>& arr) {
    for (int num : arr) {
        cout << num << " ";
    }
    cout << endl;
}

int main() {
    int n;
    cout << "Enter the number of elements: ";
    cin >> n;

    vector<int> arr(n);
    srand(time(0));
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000; // Generate random numbers between 0 and 999
    }

    cout << "Unsorted array: ";
    printArray(arr);

    string sortingAlgorithm = "Heap Sort";
    cout << "Using " << sortingAlgorithm << endl;
    auto start = high_resolution_clock::now();
    heapSort(arr);
    auto stop = high_resolution_clock::now();

    cout << "Sorted array: ";
    printArray(arr);

    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken to sort the array: " << duration.count() << " microseconds" << endl;

    cout << "Sorting algorithm used: " << sortingAlgorithm << endl;
    cout << "Time complexity of heap sort: O(n log n)" << endl;
    cout << "Space complexity of heap sort: O(1)" << endl;

    return 0;
}
