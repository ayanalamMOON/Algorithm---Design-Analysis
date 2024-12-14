#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <chrono>
#include <random>
#include <stdexcept>

using namespace std;

void printArray(const vector<int>& arr) {
    for (int num : arr) {
        cout << num << " ";
    }
    cout << endl;
}

int main() {
    try {
        int n;
        cout << "Enter the number of elements: ";
        cin >> n;

        if (n <= 0) {
            throw invalid_argument("Number of elements must be positive.");
        }

        vector<int> arr(n);
        mt19937 rng(random_device{}());
        uniform_int_distribution<int> dist(1, 10000);

        for (int& num : arr) {
            num = dist(rng);
        }

        cout << "Unsorted array: ";
        printArray(arr);

        auto start = chrono::high_resolution_clock::now();
        sort(arr.begin(), arr.end());
        auto end = chrono::high_resolution_clock::now();

        chrono::duration<double, milli> duration = end - start;

        cout << "Sorting algorithm used: IntroSort (std::sort)" << endl;
        cout << "Sorted array: ";
        printArray(arr);

        cout << "Time taken to sort: " << duration.count() << " ms" << endl;
        cout << "Time complexity: O(n log n)" << endl;
        cout << "Space complexity: O(log n)" << endl;
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }

    return 0;
}