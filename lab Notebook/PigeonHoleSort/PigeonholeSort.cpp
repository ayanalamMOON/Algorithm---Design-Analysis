#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <exception>

using namespace std;

// Function to perform pigeonhole sort
void pigeonholeSort(vector<int>& arr) {
    int min = *min_element(arr.begin(), arr.end());
    int max = *max_element(arr.begin(), arr.end());
    int range = max - min + 1;

    vector<int> holes(range, 0);

    for (int num : arr) {
        holes[num - min]++;
    }

    int index = 0;
    for (int i = 0; i < range; ++i) {
        while (holes[i]-- > 0) {
            arr[index++] = i + min;
        }
    }
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
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(0, 1000);

        cout << "Unsorted array: ";
        for (int i = 0; i < n; ++i) {
            arr[i] = dis(gen);
            cout << arr[i] << " ";
        }
        cout << endl;

        auto start = chrono::high_resolution_clock::now();
        pigeonholeSort(arr);
        auto end = chrono::high_resolution_clock::now();

        chrono::duration<double, milli> duration = end - start;

        cout << "Sorting Algorithm: Pigeonhole Sort" << endl;
        cout << "Sorted array: ";
        for (int num : arr) {
            cout << num << " ";
        }
        cout << endl;

        cout << "Time taken to sort the array: " << duration.count() << " ms" << endl;
        cout << "Time Complexity: O(n + Range)" << endl;
        cout << "Space Complexity: O(Range)" << endl;

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }

    return 0;
}
