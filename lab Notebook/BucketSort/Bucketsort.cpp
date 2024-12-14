#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include <memory>
#include <exception>

using namespace std;

void bucketSort(vector<int>& arr, int n) {
    if (n <= 0)
        return;

    // 1) Create n empty buckets
    vector<vector<int>> b(n);

    // 2) Put array elements in different buckets
    for (int i = 0; i < n; i++) {
        int bi = arr[i] / n; // Index in bucket
        b[bi].push_back(arr[i]);
    }

    // 3) Sort individual buckets
    for (int i = 0; i < n; i++)
        sort(b[i].begin(), b[i].end());

    // 4) Concatenate all buckets into arr[]
    int index = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < b[i].size(); j++)
            arr[index++] = b[i][j];
}

int main() {
    try {
        int n;
        cout << "Enter the number of elements: ";
        cin >> n;

        if (n <= 0) {
            cout << "Number of elements must be greater than 0." << endl;
            return 1;
        }

        vector<int> arr(n);
        srand(static_cast<unsigned int>(time(0)));
        for (int i = 0; i < n; i++) {
            arr[i] = rand() % 1000; // Generate random numbers between 0 and 999
        }

        cout << "Unsorted array: ";
        for (int i = 0; i < n; i++) {
            cout << arr[i] << " ";
        }
        cout << endl;

        auto start = chrono::high_resolution_clock::now();
        bucketSort(arr, n);
        auto end = chrono::high_resolution_clock::now();

        cout << "Sorting algorithm used: Bucket Sort" << endl;
        cout << "Sorted array: ";
        for (int i = 0; i < n; i++) {
            cout << arr[i] << " ";
        }
        cout << endl;

        chrono::duration<double, milli> duration = end - start;
        cout << "Time taken to sort the array: " << duration.count() << " milliseconds" << endl;
        cout << "Time complexity: O(n + k)" << endl;
        cout << "Space complexity: O(n + k)" << endl;

    } catch (const bad_alloc& e) {
        cerr << "Memory allocation failed: " << e.what() << endl;
        return 1;
    } catch (const exception& e) {
        cerr << "An error occurred: " << e.what() << endl;
        return 1;
    }

    return 0;
}
