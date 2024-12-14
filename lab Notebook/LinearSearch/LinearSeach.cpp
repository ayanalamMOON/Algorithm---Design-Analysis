#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;

void linearSearch(const vector<int>& arr, int target) {
    auto start = chrono::high_resolution_clock::now();
    bool found = false;
    for (int i = 0; i < arr.size(); ++i) {
        if (arr[i] == target) {
            found = true;
            break;
        }
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end - start;

    cout << "Searching Algorithm: Linear Search" << endl;
    if (found) {
        cout << "Element " << target << " found in the array." << endl;
    } else {
        cout << "Element " << target << " not found in the array." << endl;
    }
    cout << "Time taken to search: " << duration.count() << " milliseconds" << endl;
    cout << "Time Complexity: O(n)" << endl;
    cout << "Space Complexity: O(1)" << endl;
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
    for (int i = 0; i < n; ++i) {
        cout << arr[i] << " ";
    }
    cout << endl;

    cout << "Enter the element to search for: ";
    cin >> target;

    linearSearch(arr, target);

    return 0;
}
