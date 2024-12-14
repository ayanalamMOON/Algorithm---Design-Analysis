#include <iostream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <chrono>

using namespace std;

// ...existing code...

void countingSort(vector<int>& arr) {
    // ...existing code...
}

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
    for (int i = 0; i < n; ++i) {
        arr[i] = rand() % 100; // Generate random numbers between 0 and 99
    }

    cout << "Unsorted array: ";
    printArray(arr);

    auto start = chrono::high_resolution_clock::now();
    countingSort(arr);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    cout << "Sorting Algorithm: Counting Sort" << endl;
    cout << "Sorted array: ";
    printArray(arr);

    cout << "Time taken to sort the array: " << duration.count() << " milliseconds" << endl;
    cout << "Time Complexity: O(n + k)" << endl;
    cout << "Space Complexity: O(k)" << endl;

    return 0;
}
