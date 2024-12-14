#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Function to perform Comb Sort
void combSort(vector<int>& arr) {
    int gap = arr.size();
    bool swapped = true;

    while (gap != 1 || swapped) {
        gap = max(1, (gap * 10) / 13);
        swapped = false;

        for (int i = 0; i < arr.size() - gap; i++) {
            if (arr[i] > arr[i + gap]) {
                swap(arr[i], arr[i + gap]);
                swapped = true;
            }
        }
    }
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
    for (int i = 0; i < n; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    cout << "Sorting Algorithm: Comb Sort" << endl;

    auto start = high_resolution_clock::now();
    combSort(arr);
    auto stop = high_resolution_clock::now();

    cout << "Sorted array: ";
    for (int i = 0; i < n; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken to sort the array: " << duration.count() << " microseconds" << endl;

    cout << "Time Complexity: O(n^2 / 2^p) where p is the number of increments" << endl;
    cout << "Space Complexity: O(1)" << endl;

    return 0;
}
