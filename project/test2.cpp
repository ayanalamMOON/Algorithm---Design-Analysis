#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <iomanip>
#include <functional>
#include <map>
#include <limits>
#include <queue>
using namespace std;

struct Edge {
    int source, dest, weight;
};

struct Graph {
    int V, E;
    vector<Edge> edges;
    vector<vector<pair<int, int>>> adjacencyList;

    Graph(int vertices, int edges) : V(vertices), E(edges) {
        adjacencyList.resize(V);
    }

    void addEdge(int src, int dest, int weight) {
        edges.push_back({src, dest, weight});
        adjacencyList[src].push_back({dest, weight});
    }
};

struct ComplexityInfo {
    string timeComplexity;
    string spaceComplexity;
};

class AlgorithmPerformanceTester {
private:
    // Add RAII-based memory management
    unique_ptr<vector<int>> originalData;
    unique_ptr<vector<int>> workingData;

    // Add bounds checking
    void checkArrayBounds(size_t index, size_t size) {
        if (index >= size) {
            throw out_of_range("Array index out of bounds");
        }
    }
    
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
        {"Binary Search", {"O(log n)", "O(1)"}},
        {"Dijkstra", {"O(V log V + E)", "O(V)"}},
        {"Bellman-Ford", {"O(VE)", "O(V)"}},
        {"Floyd-Warshall", {"O(V³)", "O(V²)"}}
    };

    // Utility function to generate random data
    void generateRandomData(int size) {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> distrib(1, 10000);

        originalData->clear();
        for (int i = 0; i < size; ++i) {
            originalData->push_back(distrib(gen));
        }
        *workingData = *originalData;

        // Display generated integers
        cout << "Generated integers: ";
        for (const auto& num : *originalData) {
            cout << num << " ";
        }
        cout << "\n";
    }

    // Performance measurement template
    template<typename Func>
    double measurePerformance(Func sortingAlgorithm) {
        *workingData = *originalData;  // Reset to original data
        
        auto start = chrono::high_resolution_clock::now();
        (this->*sortingAlgorithm)(*workingData);
        auto end = chrono::high_resolution_clock::now();

        chrono::duration<double, milli> duration = end - start;
        return duration.count();
    }

    // Sorting Algorithms
    void bubbleSort(vector<int>& arr) {
        int n = arr.size();
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    swap(arr[j], arr[j + 1]);
                }
            }
        }
    }

    void selectionSort(vector<int>& arr) {
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

    void insertionSort(vector<int>& arr) {
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

    void quickSort(vector<int>& arr, int low, int high) {
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

    void mergeSort(vector<int>& arr, int left, int right) {
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

    void countSort(vector<int>& arr) {
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

    void heapSort(vector<int>& arr) {
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

    void radixSort(vector<int>& arr) {
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

    void bucketSort(vector<int>& arr) {
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

    // Add new graph algorithm implementations
    static vector<int> dijkstra(const Graph& graph, int source) {
        vector<int> dist(graph.V, numeric_limits<int>::max());
        dist[source] = 0;
        
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
        pq.push({0, source});
        
        while (!pq.empty()) {
            int u = pq.top().second;
            pq.pop();
            
            for (const auto& edge : graph.adjacencyList[u]) {
                int v = edge.first;
                int weight = edge.second;
                
                if (dist[v] > dist[u] + weight) {
                    dist[v] = dist[u] + weight;
                    pq.push({dist[v], v});
                }
            }
        }
        return dist;
    }

    static vector<int> bellmanFord(const Graph& graph, int source) {
        vector<int> dist(graph.V, numeric_limits<int>::max());
        dist[source] = 0;
        
        for (int i = 1; i <= graph.V - 1; i++) {
            for (const auto& edge : graph.edges) {
                if (dist[edge.source] != numeric_limits<int>::max() && 
                    dist[edge.source] + edge.weight < dist[edge.dest]) {
                    dist[edge.dest] = dist[edge.source] + edge.weight;
                }
            }
        }
        return dist;
    }

    // Add Floyd-Warshall implementation
    static vector<vector<int>> floydWarshall(const Graph& graph) {
        vector<vector<int>> dist(graph.V, vector<int>(graph.V, numeric_limits<int>::max()));
        
        // Initialize distances
        for (int i = 0; i < graph.V; i++) {
            dist[i][i] = 0;
            for (const auto& edge : graph.adjacencyList[i]) {
                dist[i][edge.first] = edge.second;
            }
        }
        
        // Floyd-Warshall algorithm
        for (int k = 0; k < graph.V; k++) {
            for (int i = 0; i < graph.V; i++) {
                for (int j = 0; j < graph.V; j++) {
                    if (dist[i][k] != numeric_limits<int>::max() && 
                        dist[k][j] != numeric_limits<int>::max() && 
                        dist[i][k] + dist[k][j] < dist[i][j]) {
                        dist[i][j] = dist[i][k] + dist[k][j];
                    }
                }
            }
        }
        return dist;
    }

public:
    void runPerformanceTest() {
        int size;
        cout << "Enter the number of elements to generate: ";
        cin >> size;

        // Generate random data
        generateRandomData(size);

        // Sorting Performance Tests
        vector<pair<string, void(AlgorithmPerformanceTester::*)(vector<int>&)>> sortingAlgorithms = {
            {"Bubble Sort", &AlgorithmPerformanceTester::bubbleSort},
            {"Selection Sort", &AlgorithmPerformanceTester::selectionSort},
            {"Insertion Sort", &AlgorithmPerformanceTester::insertionSort},
            {"Quick Sort", &AlgorithmPerformanceTester::quickSortWrapper},
            {"Merge Sort", &AlgorithmPerformanceTester::mergeSortWrapper},
            {"Heap Sort", &AlgorithmPerformanceTester::heapSort},
            {"Count Sort", &AlgorithmPerformanceTester::countSort},
            {"Radix Sort", &AlgorithmPerformanceTester::radixSort},
            {"Bucket Sort", &AlgorithmPerformanceTester::bucketSort},
            {"C++ Standard Sort", nullptr}
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
        sort(workingData->begin(), workingData->end());

        // Display sorted integers
        cout << "Sorted integers for searching: ";
        for (const auto& num : *workingData) {
            cout << num << " ";
        }
        cout << "\n";  // Remove the stray '2' character that was here

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
            int result = algo.second(*workingData, searchTarget);
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

    void runGraphAlgorithms() {
        cout << "\n--- Graph Algorithm Testing ---\n";
        
        int V, E;
        cout << "Enter number of vertices: ";
        cin >> V;
        cout << "Enter number of edges: ";
        cin >> E;
        
        Graph graph(V, E);
        
        cout << "Enter edge information (source destination weight):\n";
        for (int i = 0; i < E; i++) {
            int src, dest, weight;
            cin >> src >> dest >> weight;
            graph.addEdge(src, dest, weight);
        }
        
        int source;
        cout << "Enter source vertex: ";
        cin >> source;

        cout << "\n--- Graph Algorithm Performance ---\n";
        cout << setw(20) << "Algorithm" 
             << setw(15) << "Time (ms)"
             << setw(20) << "Time Complexity"
             << setw(20) << "Space Complexity\n";
        cout << string(75, '-') << "\n";

        // Test Dijkstra's Algorithm
        auto start = chrono::high_resolution_clock::now();
        vector<int> dijkstraResult = dijkstra(graph, source);
        auto end = chrono::high_resolution_clock::now();
        double dijkstraTime = chrono::duration<double, milli>(end - start).count();

        cout << setw(20) << "Dijkstra"
             << setw(15) << fixed << setprecision(4) << dijkstraTime
             << setw(20) << complexityMap["Dijkstra"].timeComplexity
             << setw(20) << complexityMap["Dijkstra"].spaceComplexity << "\n";

        // Test Bellman-Ford Algorithm
        start = chrono::high_resolution_clock::now();
        vector<int> bellmanFordResult = bellmanFord(graph, source);
        end = chrono::high_resolution_clock::now();
        double bellmanFordTime = chrono::duration<double, milli>(end - start).count();

        cout << setw(20) << "Bellman-Ford"
             << setw(15) << fixed << setprecision(4) << bellmanFordTime
             << setw(20) << complexityMap["Bellman-Ford"].timeComplexity
             << setw(20) << complexityMap["Bellman-Ford"].spaceComplexity << "\n";

        // Add Floyd-Warshall testing after Bellman-Ford
        start = chrono::high_resolution_clock::now();
        vector<vector<int>> floydWarshallResult = floydWarshall(graph);
        end = chrono::high_resolution_clock::now();
        double floydWarshallTime = chrono::duration<double, milli>(end - start).count();

        cout << setw(20) << "Floyd-Warshall"
             << setw(15) << fixed << setprecision(4) << floydWarshallTime
             << setw(20) << complexityMap["Floyd-Warshall"].timeComplexity
             << setw(20) << complexityMap["Floyd-Warshall"].spaceComplexity << "\n";

        // Print results
        cout << "\nShortest distances from vertex " << source << ":\n";
        for (int i = 0; i < V; i++) {
            cout << "To vertex " << i << ": ";
            cout << "Dijkstra: " << (dijkstraResult[i] == numeric_limits<int>::max() ? "INF" : to_string(dijkstraResult[i]));
            cout << ", Bellman-Ford: " << (bellmanFordResult[i] == numeric_limits<int>::max() ? "INF" : to_string(bellmanFordResult[i])) << "\n";
        }

        // Update results printing
        cout << "\nAll-pairs shortest paths:\n";
        for (int i = 0; i < graph.V; i++) {
            cout << "\nFrom vertex " << i << ":\n";
            for (int j = 0; j < graph.V; j++) {
                if (i != j) {
                    cout << "To vertex " << j << ": ";
                    if (i == source) {
                        cout << "Dijkstra: " 
                             << (dijkstraResult[j] == numeric_limits<int>::max() ? "INF" : to_string(dijkstraResult[j]))
                             << ", Bellman-Ford: "
                             << (bellmanFordResult[j] == numeric_limits<int>::max() ? "INF" : to_string(bellmanFordResult[j]));
                    }
                    cout << ", Floyd-Warshall: "
                         << (floydWarshallResult[i][j] == numeric_limits<int>::max() ? "INF" : to_string(floydWarshallResult[i][j]))
                         << "\n";
                }
            }
        }
    }

    // Add wrapper functions for quickSort and mergeSort
    void quickSortWrapper(vector<int>& arr) {
        quickSort(arr, 0, arr.size() - 1);
    }

    void mergeSortWrapper(vector<int>& arr) {
        mergeSort(arr, 0, arr.size() - 1);
    }
};

int main() {
    AlgorithmPerformanceTester tester;
    
    int choice;
    do {
        cout << "\n1. Test Sorting and Searching Algorithms\n";
        cout << "2. Test Graph Algorithms\n";
        cout << "3. Exit\n";
        cout << "Enter your choice: ";
        cin >> choice;

        switch (choice) {
            case 1:
                tester.runPerformanceTest();
                break;
            case 2:
                tester.runGraphAlgorithms();
                break;
            case 3:
                cout << "Exiting...\n";
                break;
            default:
                cout << "Invalid choice!\n";
        }
    } while (choice != 3);

    return 0;
}