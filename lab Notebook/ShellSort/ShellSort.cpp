#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;

void shellSort(vector<int>& arr) {
    int n = arr.size();
    for (int gap = n / 2; gap > 0; gap /= 2) {
        for (int i = gap; i < n; i++) {
            int temp = arr[i];
            int j;
            for (j = i; j >= gap && arr[j - gap] > temp; j -= gap) {
                arr[j] = arr[j - gap];
            }
            arr[j] = temp;
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

    auto start = chrono::high_resolution_clock::now();
    shellSort(arr);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end - start; // Change to milliseconds

    cout << "Sorted array: ";
    for (int i = 0; i < n; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    cout << "Time taken to sort the array: " << duration.count() << " milliseconds" << endl;
    cout << "Time complexity: O(n^1.5)" << endl;
    cout << "Space complexity: O(1)" << endl;
    cout << "Sorting algorithm used: Shell Sort" << endl;

    return 0;
}
