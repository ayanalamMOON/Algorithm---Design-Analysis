#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <chrono>

using namespace std;

// Function to print the array
void printArray(const vector<int>& arr) {
    for (int num : arr) {
        cout << num << " ";
    }
    cout << endl;
}

// Function to get the maximum value in the array
int getMax(const vector<int>& arr) {
    int max = arr[0];
    for (int num : arr) {
        if (num > max) {
            max = num;
        }
    }
    return max;
}

// Function to perform counting sort based on the digit represented by exp
void countingSort(vector<int>& arr, int exp) {
    int n = arr.size();
    vector<int> output(n);
    int count[10] = {0};

    for (int i = 0; i < n; i++) {
        count[(arr[i] / exp) % 10]++;
    }

    for (int i = 1; i < 10; i++) {
        count[i] += count[i - 1];
    }

    for (int i = n - 1; i >= 0; i--) {
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        count[(arr[i] / exp) % 10]--;
    }

    for (int i = 0; i < n; i++) {
        arr[i] = output[i];
    }
}

// Function to perform radix sort
void radixSort(vector<int>& arr) {
    int max = getMax(arr);
    for (int exp = 1; max / exp > 0; exp *= 10) {
        countingSort(arr, exp);
    }
}

int main() {
    int n;
    cout << "Enter the number of elements: ";
    cin >> n;

    vector<int> arr(n);
    srand(time(0));
    for (int i = 0; i < n; ++i) {
        arr[i] = rand() % 1000; // Generate random numbers between 0 and 999
    }

    cout << "Unsorted array: ";
    printArray(arr);

    auto start = chrono::high_resolution_clock::now();
    radixSort(arr);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> duration = end - start;

    cout << "Sorting algorithm used: Radix Sort" << endl;
    cout << "Sorted array: ";
    printArray(arr);
    cout << "Time taken to sort the array: " << duration.count() << " ms" << endl;
    cout << "Time complexity: O(d*(n+b)) where d is the number of digits, n is the number of elements, and b is the base" << endl;
    cout << "Space complexity: O(n+b)" << endl;

    return 0;
}
