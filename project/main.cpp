#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <iomanip>
#include <functional>
#include <map>

using namespace std;

struct ComplexityInfo {
    string timeComplexity;
    string spaceComplexity;
};

class AlgorithmPerformanceTester {
private:
    vector<int> originalData;
    vector<int> workingData;
    
    // Add complexity information for each algorithm
    map<string, ComplexityInfo> complexityMap = {
        {"Bubble Sort", {"O(n²)", "O(1)"}},
        {"Selection Sort", {"O(n²)", "O(1)"}},
        {"Insertion Sort", {"O(n²)", "O(1)"}},
        {"Quick Sort", {"O(n log n)", "O(log n)"}},
        {"Merge Sort", {"O(n log n)", "O(n)"}},
        {"Heap Sort", {"O(n log n)", "O(1)"}},
        {"Count Sort", {"O(n + k)", "O(k)"}},
        {"Radix Sort", {"O(d * (n + k))", "O(n + k)"}},
        {"Bucket Sort", {"O(n + k)", "O(n)"}},
        {"C++ Standard Sort", {"O(n log n)", "O(log n)"}},
        {"Linear Search", {"O(n)", "O(1)"}},
        {"Binary Search", {"O(log n)", "O(1)"}}
    };

    // Utility function to generate random data
    void generateRandomData(int size) {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> distrib(1, 10000);

        originalData.clear();
        for (int i = 0; i < size; ++i) {
            originalData.push_back(distrib(gen));
        }
        workingData = originalData;

        // Display generated integers
        cout << "Generated integers: ";
        for (const auto& num : originalData) {
            cout << num << " ";
        }
        cout << "\n";
    }

    // Performance measurement template
    template<typename Func>
    double measurePerformance(Func sortingAlgorithm) {
        workingData = originalData;  // Reset to original data
        
        auto start = chrono::high_resolution_clock::now();
        sortingAlgorithm(workingData);
        auto end = chrono::high_resolution_clock::now();

        chrono::duration<double, milli> duration = end - start;
        return duration.count();
    }

    // Sorting Algorithms
    static void bubbleSort(vector<int>& arr) {
        int n = arr.size();
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    swap(arr[j], arr[j + 1]);
                }
            }
        }
    }

    static void selectionSort(vector<int>& arr) {
        int n = arr.size();
        for (int i = 0; i < n - 1; i++) {
            int min_idx = i;
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < arr[min_idx]) {
                    min_idx = j;
                }
            }
            swap(arr[i], arr[min_idx]);
        }
    }

    static void insertionSort(vector<int>& arr) {
        int n = arr.size();
        for (int i = 1; i < n; i++) {
            int key = arr[i];
            int j = i - 1;
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
    }

    static void quickSort(vector<int>& arr, int low, int high) {
        if (low < high) {
            int pivot = arr[high];
            int i = low - 1;

            for (int j = low; j <= high - 1; j++) {
                if (arr[j] < pivot) {
                    i++;
                    swap(arr[i], arr[j]);
                }
            }
            swap(arr[i + 1], arr[high]);
            int partitionIndex = i + 1;

            quickSort(arr, low, partitionIndex - 1);
            quickSort(arr, partitionIndex + 1, high);
        }
    }

    static void mergeSort(vector<int>& arr, int left, int right) {
        if (left >= right) return;

        int mid = left + (right - left) / 2;
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        vector<int> temp(right - left + 1);
        int i = left, j = mid + 1, k = 0;

        while (i <= mid && j <= right) {
            if (arr[i] <= arr[j]) {
                temp[k++] = arr[i++];
            } else {
                temp[k++] = arr[j++];
            }
        }

        while (i <= mid) temp[k++] = arr[i++];
        while (j <= right) temp[k++] = arr[j++];

        for (int p = 0; p < k; p++) {
            arr[left + p] = temp[p];
        }
    }

    static void countSort(vector<int>& arr) {
        int max = *max_element(arr.begin(), arr.end());
        vector<int> count(max + 1, 0);
        vector<int> output(arr.size());

        for(int i : arr) count[i]++;
        for(int i = 1; i <= max; i++) count[i] += count[i-1];

        for(int i = arr.size()-1; i >= 0; i--) {
            output[count[arr[i]]-1] = arr[i];
            count[arr[i]]--;
        }
        arr = output;
    }

    static void heapSort(vector<int>& arr) {
        function<void(vector<int>&, int, int)> heapify = 
            [&heapify](vector<int>& arr, int n, int i) {
            int largest = i;
            int left = 2 * i + 1;
            int right = 2 * i + 2;

            if (left < n && arr[left] > arr[largest]) largest = left;
            if (right < n && arr[right] > arr[largest]) largest = right;

            if (largest != i) {
                swap(arr[i], arr[largest]);
                heapify(arr, n, largest);
            }
        };

        int n = arr.size();
        for (int i = n / 2 - 1; i >= 0; i--)
            heapify(arr, n, i);

        for (int i = n - 1; i > 0; i--) {
            swap(arr[0], arr[i]);
            heapify(arr, i, 0);
        }
    }

    static void radixSort(vector<int>& arr) {
        int max = *max_element(arr.begin(), arr.end());
        
        for (int exp = 1; max/exp > 0; exp *= 10) {
            vector<int> output(arr.size());
            vector<int> count(10, 0);

            for (int i : arr) count[(i/exp)%10]++;
            for (int i = 1; i < 10; i++) count[i] += count[i-1];

            for (int i = arr.size()-1; i >= 0; i--) {
                output[count[(arr[i]/exp)%10]-1] = arr[i];
                count[(arr[i]/exp)%10]--;
            }
            arr = output;
        }
    }

    static void bucketSort(vector<int>& arr) {
        int max = *max_element(arr.begin(), arr.end());
        int min = *min_element(arr.begin(), arr.end());
        int range = (max - min) / arr.size() + 1;
        
        vector<vector<int>> buckets(arr.size());

        for (int i : arr) {
            int index = (i - min) / range;
            buckets[index].push_back(i);
        }

        for (auto& bucket : buckets) {
            sort(bucket.begin(), bucket.end());
        }

        int index = 0;
        for (const auto& bucket : buckets) {
            for (int value : bucket) {
                arr[index++] = value;
            }
        }
    }

    // Searching Algorithms
    static int linearSearch(const vector<int>& arr, int target) {
        for (size_t i = 0; i < arr.size(); i++) {
            if (arr[i] == target) return i;
        }
        return -1;
    }

    static int binarySearch(vector<int>& arr, int target) {
        int left = 0, right = arr.size() - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (arr[mid] == target) return mid;
            if (arr[mid] < target) left = mid + 1;
            else right = mid - 1;
        }
        return -1;
    }

public:
    void runPerformanceTest() {
        int size;
        cout << "Enter the number of elements to generate: ";
        cin >> size;

        // Generate random data
        generateRandomData(size);

        // Sorting Performance Tests
        vector<pair<string, function<void(vector<int>&)>>> sortingAlgorithms = {
            {"Bubble Sort", [this](vector<int>& arr) { bubbleSort(arr); }},
            {"Selection Sort", [this](vector<int>& arr) { selectionSort(arr); }},
            {"Insertion Sort", [this](vector<int>& arr) { insertionSort(arr); }},
            {"Quick Sort", [this](vector<int>& arr) { quickSort(arr, 0, arr.size() - 1); }},
            {"Merge Sort", [this](vector<int>& arr) { mergeSort(arr, 0, arr.size() - 1); }},
            {"Heap Sort", [this](vector<int>& arr) { heapSort(arr); }},
            {"Count Sort", [this](vector<int>& arr) { countSort(arr); }},
            {"Radix Sort", [this](vector<int>& arr) { radixSort(arr); }},
            {"Bucket Sort", [this](vector<int>& arr) { bucketSort(arr); }},
            {"C++ Standard Sort", [](vector<int>& arr) { sort(arr.begin(), arr.end()); }}
        };

        cout << "\n--- Sorting Algorithm Performance ---\n";
        cout << setw(20) << "Algorithm" 
                  << setw(15) << "Time (ms)"
                  << setw(20) << "Time Complexity"
                  << setw(20) << "Space Complexity\n";
        cout << string(75, '-') << "\n";

        vector<pair<string, double>> sortingTimes;

        for (const auto& algo : sortingAlgorithms) {
            double time = measurePerformance(algo.second);
            sortingTimes.push_back({algo.first, time});
            
            const auto& complexity = complexityMap[algo.first];
            cout << fixed << setprecision(4)
                      << setw(20) << algo.first
                      << setw(15) << time
                      << setw(20) << complexity.timeComplexity
                      << setw(20) << complexity.spaceComplexity << "\n";
        }

        // Find fastest sorting algorithm
        auto fastestSort = *min_element(sortingTimes.begin(), sortingTimes.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        cout << "\nFastest Sorting Algorithm: " << fastestSort.first 
                  << " (" << fastestSort.second << " ms)\n";

        // Sort the array first for binary search
        sort(workingData.begin(), workingData.end());

        // Display sorted integers
        cout << "Sorted integers for searching: ";
        for (const auto& num : workingData) {
            cout << num << " ";
        }
        cout << "\n";

        int searchTarget;
        cout << "\nEnter a value to search: ";
        cin >> searchTarget;

        vector<pair<string, function<int(const vector<int>&, int)>>> searchAlgorithms = {
            {"Linear Search", linearSearch},
            {"Binary Search", [this](const vector<int>& arr, int target) { 
                return binarySearch(const_cast<vector<int>&>(arr), target); 
            }}
        };

        cout << "\n--- Searching Algorithm Performance ---\n";
        cout << setw(20) << "Algorithm" 
                  << setw(15) << "Time (ms)"
                  << setw(20) << "Time Complexity"
                  << setw(20) << "Space Complexity"
                  << setw(15) << "Result\n";
        cout << string(90, '-') << "\n";

        vector<pair<string, double>> searchTimes;

        for (const auto& algo : searchAlgorithms) {
            auto start = chrono::high_resolution_clock::now();
            int result = algo.second(workingData, searchTarget);
            auto end = chrono::high_resolution_clock::now();

            chrono::duration<double, milli> duration = end - start;
            double time = duration.count();

            searchTimes.push_back({algo.first, time});
            const auto& complexity = complexityMap[algo.first];
            
            cout << fixed << setprecision(4)
                      << setw(20) << algo.first
                      << setw(15) << time
                      << setw(20) << complexity.timeComplexity
                      << setw(20) << complexity.spaceComplexity
                      << setw(15) << (result != -1 ? "Found" : "Not Found") << "\n";
        }

        // Find fastest search algorithm
        auto fastestSearch = *min_element(searchTimes.begin(), searchTimes.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        cout << "\nFastest Searching Algorithm: " << fastestSearch.first 
                  << " (" << fastestSearch.second << " ms)\n";
    }
};

int main() {
    AlgorithmPerformanceTester tester;
    tester.runPerformanceTest();
    return 0;
}