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

int main() {
    int n;
    cout << "Enter the number of elements: ";
    cin >> n;

    vector<int> arr(n);
    srand(time(0));
    // Generate random numbers between 0 and 999
    for (int i = 0; i < n; ++i) {
        arr[i] = rand() % 1000;
    }

    cout << "Unsorted array: ";
    printArray(arr);

    // Measure the time taken to sort the array
    auto start = chrono::high_resolution_clock::now();
    sort(arr.begin(), arr.end()); // Using std::sort which uses TimSort in most implementations
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> duration = end - start;

    cout << "Sorted array: ";
    printArray(arr);

    // Print sorting algorithm details and time taken
    cout << "Sorting algorithm used: TimSort (via std::sort)" << endl;
    cout << "Time taken to sort the array: " << duration.count() << " seconds" << endl;
    cout << "Time complexity: O(n log n)" << endl;
    cout << "Space complexity: O(n)" << endl;

    return 0;
}
