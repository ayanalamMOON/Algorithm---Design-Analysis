#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;

int binarySearch(const vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

int main() {
    srand(time(0));
    int n, target;
    cout << "Enter the number of elements in the array: ";
    cin >> n;

    vector<int> arr(n);
    for (int i = 0; i < n; ++i) {
        arr[i] = rand() % 1000; // Generate random numbers between 0 and 999
    }
    sort(arr.begin(), arr.end());

    cout << "Generated sorted array: ";
    for (int num : arr) {
        cout << num << " ";
    }
    cout << endl;

    cout << "Enter the element to search for: ";
    cin >> target;

    auto start = chrono::high_resolution_clock::now();
    int result = binarySearch(arr, target);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end - start;

    cout << "Searching algorithm used: Binary Search" << endl;
    if (result != -1) {
        cout << "Element found at index: " << result << endl;
    } else {
        cout << "Element not found" << endl;
    }
    cout << "Time taken to search: " << duration.count() << " milliseconds" << endl;
    cout << "Time complexity: O(log n)" << endl;
    cout << "Space complexity: O(1)" << endl;

    return 0;
}
