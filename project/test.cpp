/**
 * @file test.cpp
 * @brief Algorithm Performance Testing Framework
 * @author 
 * @version 1.1
 * @date 2024-01-20
 * 
 * This program implements and analyzes various algorithms including:
 * - Sorting algorithms
 * - Searching algorithms
 * - Graph algorithms
 * - Dynamic programming
 * - String matching algorithms
 */

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
#include <stack>  // Add this header for stack implementation
#include <fstream>
#include <ctime>
#include <sstream>

#ifdef _WIN32
    #include <windows.h>
    #include <psapi.h>
    #pragma comment(lib, "psapi.lib")
#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    #include <unistd.h>
    #include <sys/resource.h>
    #if defined(__APPLE__) && defined(__MACH__)
        #include <mach/mach.h>
    #elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
        #include <fcntl.h>
        #include <procfs.h>
    #else
        #include <sys/sysinfo.h>
    #endif
#endif

using namespace std;

// Configuration constants
namespace Config {
    constexpr int kMaxVertices = 1000;
    constexpr int kMaxDataSize = 1000000;
    constexpr int kMinDataSize = 1;
    constexpr double kTimeoutSeconds = 60.0;
    constexpr int kBenchmarkIterations = 5;
}

// Logger class for performance metrics
class Logger {
public:
    static void LogPerformance(const std::string& algorithm, double time_ms, 
                            const std::string& complexity) {
        std::ofstream log_file("algorithm_performance.log", std::ios::app);
        if (log_file.is_open()) {
            std::time_t now = std::time(nullptr);
            log_file << std::ctime(&now) 
                    << algorithm << ": " 
                    << time_ms << "ms, "
                    << complexity << std::endl;
        }
    }

    static void LogError(const std::string& message) {
        std::ofstream log_file("error.log", std::ios::app);
        if (log_file.is_open()) {
            std::time_t now = std::time(nullptr);
            log_file << std::ctime(&now) << message << std::endl;
        }
    }
};

// Input validator class
class InputValidator {
public:
    static bool ValidateSize(int size) {
        return size >= Config::kMinDataSize && size <= Config::kMaxDataSize;
    }

    static bool ValidateGraphInput(int vertices, int edges) {
        return vertices > 0 && vertices <= Config::kMaxVertices &&
               edges >= 0 && edges <= vertices * (vertices - 1);
    }
};

// Algorithm result structure
struct AlgorithmResult {
    double execution_time;
    std::string algorithm_name;
    std::string complexity;
    bool success;
    std::string error_message;
};

// Performance metrics visualization
class PerformanceVisualizer {
public:
    static void DisplayBarChart(const std::vector<AlgorithmResult>& results) {
        const int width = 50; // Maximum bar width
        double max_time = 0;
        
        for (const auto& result : results) {
            max_time = std::max(max_time, result.execution_time);
        }

        for (const auto& result : results) {
            int bar_length = static_cast<int>(
                (result.execution_time / max_time) * width);
            
            std::cout << std::setw(20) << std::left << result.algorithm_name 
                     << " |" << std::string(bar_length, '#') 
                     << " " << result.execution_time << "ms\n";
        }
    }
};

struct Edge {
    int source, dest, weight;
};

// Update Graph structure with better error handling
struct Graph {
    int V, E;
    vector<Edge> edges;
    vector<vector<pair<int, int>>> adjacencyList;
    vector<vector<int>> grid;  // For A* pathfinding

    Graph(int vertices, int edges) : V(vertices), E(edges) {
        adjacencyList.resize(V);
        grid.resize(V, vector<int>(V, 1)); // Initialize grid for A*
    }

    void ValidateVertex(int v) const {
        if (v < 0 || v >= V) {
            throw std::out_of_range("Vertex index out of range");
        }
    }

    void addEdge(int src, int dest, int weight) {
        try {
            ValidateVertex(src);
            ValidateVertex(dest);
            if (weight < 0) {
                throw std::invalid_argument("Negative weight not allowed");
            }
            edges.push_back({src, dest, weight});
            adjacencyList[src].push_back({dest, weight});
            // For undirected graph algorithms (MST)
            adjacencyList[dest].push_back({src, weight});
        } catch (const std::exception& e) {
            Logger::LogError("Edge addition failed: " + std::string(e.what()));
            throw;
        }
    }
};

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
        {"Binary Search", {"O(log n)", "O(1)"}},
        {"Dijkstra", {"O(V log V + E)", "O(V)"}},
        {"Bellman-Ford", {"O(VE)", "O(V)"}},
        {"Floyd-Warshall", {"O(V³)", "O(V²)"}},
        {"Prim", {"O(E log V)", "O(V)"}},
        {"Kruskal", {"O(E log E)", "O(V)"}},
        {"A*", {"O(E)", "O(V)"}},
        {"Topological Sort", {"O(V + E)", "O(V)"}},
        {"Knapsack", {"O(nW)", "O(nW)"}},
        {"LIS", {"O(n²)", "O(n)"}},
        {"Matrix Chain", {"O(n³)", "O(n²)"}},
        {"KMP", {"O(n + m)", "O(m)"}},
        {"Rabin-Karp", {"O(n + m)", "O(1)"}},
        {"DFS", {"O(V + E)", "O(V)"}},
        {"BFS", {"O(V + E)", "O(V)"}},
        {"Ford-Fulkerson", {"O(E * max_flow)", "O(V)"}},
        {"Edmonds-Karp", {"O(VE^2)", "O(V)"}}
    };

    // Utility function to generate random data
    void generateRandomData(int size) {
        if (size <= 0 || size > 1000000) {  // Add reasonable limits
            throw runtime_error("Invalid size: must be between 1 and 1000000");
        }

        try {
            originalData.clear();
            originalData.reserve(size);  // Pre-allocate memory
            workingData.reserve(size);

            random_device rd;
            mt19937 gen(rd());
            uniform_int_distribution<> distrib(1, 10000);

            for (int i = 0; i < size; ++i) {
                originalData.push_back(distrib(gen));
            }
            workingData = originalData;

            cout << "Generated integers: ";
            for (size_t i = 0; i < min(size_t(10), originalData.size()); ++i) {
                cout << originalData[i] << " ";
            }
            if (size > 10) cout << "...";
            cout << "\n";
        } catch (const std::bad_alloc& e) {
            throw runtime_error("Memory allocation failed: " + string(e.what()));
        }
    }

    // Performance measurement template
    template<typename Func>
    double measurePerformance(Func sortingAlgorithm) {
        if (originalData.empty()) {
            throw runtime_error("No data to process");
        }

        try {
            workingData = originalData;
        } catch (const std::bad_alloc& e) {
            throw runtime_error("Memory allocation failed during copy: " + string(e.what()));
        }

        auto start = chrono::high_resolution_clock::now();
        try {
            sortingAlgorithm(workingData);
        } catch (const exception& e) {
            throw runtime_error("Algorithm execution failed: " + string(e.what()));
        }
        auto end = chrono::high_resolution_clock::now();

        return chrono::duration<double, milli>(end - start).count();
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

    // Add new graph algorithm implementations
    static vector<int> dijkstra(const Graph& graph, int source) {
        if (source < 0 || source >= graph.V) {
            throw runtime_error("Invalid source vertex");
        }

        vector<int> dist(graph.V, numeric_limits<int>::max());
        dist[source] = 0;
        
        try {
            priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
            pq.push({0, source});
            
            while (!pq.empty()) {
                int u = pq.top().second;
                pq.pop();
                
                for (const auto& edge : graph.adjacencyList[u]) {
                    int v = edge.first;
                    int weight = edge.second;
                    
                    if (v < 0 || v >= graph.V) continue;  // Skip invalid vertices
                    
                    // Check for integer overflow
                    if (dist[u] != numeric_limits<int>::max() &&
                        weight != numeric_limits<int>::max() &&
                        dist[u] + weight < dist[v]) {
                        dist[v] = dist[u] + weight;
                        pq.push({dist[v], v});
                    }
                }
            }
        } catch (const std::bad_alloc& e) {
            throw runtime_error("Memory allocation failed in Dijkstra: " + string(e.what()));
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

    // New algorithm implementations
    static vector<Edge> primMST(const Graph& graph) {
        vector<Edge> mst;
        vector<bool> visited(graph.V, false);
        vector<int> key(graph.V, INT_MAX);
        vector<int> parent(graph.V, -1);
        
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
        key[0] = 0;
        pq.push({0, 0});

        while (!pq.empty()) {
            int u = pq.top().second;
            pq.pop();
            
            if (visited[u]) continue;
            visited[u] = true;

            for (const auto& [v, weight] : graph.adjacencyList[u]) {
                if (!visited[v] && weight < key[v]) {
                    parent[v] = u;
                    key[v] = weight;
                    pq.push({key[v], v});
                }
            }
        }

        for (int i = 1; i < graph.V; i++) {
            if (parent[i] != -1) {
                mst.push_back({parent[i], i, key[i]});
            }
        }
        return mst;
    }

    static vector<Edge> kruskalMST(const Graph& graph) {
        vector<Edge> mst;
        vector<int> parent(graph.V);
        vector<int> rank(graph.V, 0);
        
        // Initialize parent array
        for (int i = 0; i < graph.V; i++) {
            parent[i] = i;
        }

        // Recursive find function with path compression
        function<int(int)> findSet = [&parent, &findSet](int i) -> int {
            if (parent[i] != i) {
                parent[i] = findSet(parent[i]);
            }
            return parent[i];
        };

        // Union function
        auto unionSets = [&findSet, &parent](int i, int j) {
            parent[findSet(i)] = findSet(j);
        };

        // Sort edges by weight
        vector<Edge> edges = graph.edges;
        sort(edges.begin(), edges.end(),
             [](const Edge& a, const Edge& b) { return a.weight < b.weight; });

        // Process each edge
        for (const Edge& edge : edges) {
            if (findSet(edge.source) != findSet(edge.dest)) {
                mst.push_back(edge);
                unionSets(edge.source, edge.dest);
            }
        }
        
        return mst;
    }

    static vector<int> astar(const Graph& graph, int start, int goal) {
        struct Node {
            int vertex;
            int g_cost;
            int f_cost;
            Node(int v, int g, int f) : vertex(v), g_cost(g), f_cost(f) {}
        };

        auto compare = [](const Node& a, const Node& b) { 
            return a.f_cost > b.f_cost; 
        };

        vector<int> path;
        vector<bool> closed(graph.V, false);
        vector<int> parent(graph.V, -1);
        priority_queue<Node, vector<Node>, decltype(compare)> open(compare);

        open.push(Node(start, 0, 0));

        while (!open.empty()) {
            Node current = open.top();
            open.pop();

            if (current.vertex == goal) {
                int curr = goal;
                while (curr != -1) {
                    path.push_back(curr);
                    curr = parent[curr];
                }
                reverse(path.begin(), path.end());
                return path;
            }

            closed[current.vertex] = true;

            for (const auto& [next, weight] : graph.adjacencyList[current.vertex]) {
                if (closed[next]) continue;

                int g_cost = current.g_cost + weight;
                int h_cost = abs(next - goal); // Simple heuristic
                int f_cost = g_cost + h_cost;

                open.push(Node(next, g_cost, f_cost));
                parent[next] = current.vertex;
            }
        }
        return path;
    }

    static vector<int> topologicalSort(const Graph& graph) {
        vector<int> result;
        vector<bool> visited(graph.V, false);
        
        function<void(int)> dfs = [&](int v) {
            visited[v] = true;
            for (const auto& [u, _] : graph.adjacencyList[v]) {
                if (!visited[u]) dfs(u);
            }
            result.push_back(v);
        };

        for (int i = 0; i < graph.V; i++) {
            if (!visited[i]) dfs(i);
        }
        
        reverse(result.begin(), result.end());
        return result;
    }

    // Add new Dynamic Programming algorithms
    static pair<int, vector<int>> knapsack(const vector<int>& values, const vector<int>& weights, int capacity) {
        int n = values.size();
        vector<vector<int>> dp(n + 1, vector<int>(capacity + 1, 0));
        
        for (int i = 1; i <= n; i++) {
            for (int w = 0; w <= capacity; w++) {
                if (weights[i-1] <= w) {
                    dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1]);
                } else {
                    dp[i][w] = dp[i-1][w];
                }
            }
        }

        // Reconstruct solution
        vector<int> selected;
        int w = capacity;
        for (int i = n; i > 0 && w > 0; i--) {
            if (dp[i][w] != dp[i-1][w]) {
                selected.push_back(i-1);
                w -= weights[i-1];
            }
        }
        return {dp[n][capacity], selected};
    }

    static vector<int> longestIncreasingSubsequence(const vector<int>& arr) {
        int n = arr.size();
        vector<int> dp(n, 1), prev(n, -1);
        int maxLen = 1, endIndex = 0;

        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (arr[i] > arr[j] && dp[j] + 1 > dp[i]) {
                    dp[i] = dp[j] + 1;
                    prev[i] = j;
                    if (dp[i] > maxLen) {
                        maxLen = dp[i];
                        endIndex = i;
                    }
                }
            }
        }

        vector<int> sequence;
        while (endIndex != -1) {
            sequence.push_back(arr[endIndex]);
            endIndex = prev[endIndex];
        }
        reverse(sequence.begin(), sequence.end());
        return sequence;
    }

    static pair<int, vector<int>> matrixChainMultiplication(const vector<int>& dimensions) {
        int n = dimensions.size() - 1;
        vector<vector<int>> dp(n, vector<int>(n, 0));
        vector<vector<int>> split(n, vector<int>(n, 0));

        for (int len = 2; len <= n; len++) {
            for (int i = 0; i < n - len + 1; i++) {
                int j = i + len - 1;
                dp[i][j] = INT_MAX;
                for (int k = i; k < j; k++) {
                    int cost = dp[i][k] + dp[k+1][j] + dimensions[i]*dimensions[k+1]*dimensions[j+1];
                    if (cost < dp[i][j]) {
                        dp[i][j] = cost;
                        split[i][j] = k;
                    }
                }
            }
        }

        // Reconstruct optimal parenthesization
        function<vector<int>(int,int)> getOrder = [&](int i, int j) -> vector<int> {
            if (i == j) return {i};
            vector<int> left = getOrder(i, split[i][j]);
            vector<int> right = getOrder(split[i][j] + 1, j);
            left.insert(left.end(), right.begin(), right.end());
            return left;
        };

        return {dp[0][n-1], getOrder(0, n-1)};
    }

    // Add String Matching algorithms
    static vector<int> computeLPSArray(const string& pattern) {
        int m = pattern.length();
        vector<int> lps(m, 0);
        int len = 0, i = 1;
        
        while (i < m) {
            if (pattern[i] == len) {
                lps[i++] = ++len;
            } else {
                if (len != 0) len = lps[len - 1];
                else lps[i++] = 0;
            }
        }
        return lps;
    }

    static vector<int> KMPSearch(const string& text, const string& pattern) {
        vector<int> matches;
        vector<int> lps = computeLPSArray(pattern);
        int n = text.length(), m = pattern.length();
        int i = 0, j = 0;
        
        while (i < n) {
            if (pattern[j] == text[i]) { i++; j++; }
            if (j == m) {
                matches.push_back(i - j);
                j = lps[j - 1];
            } else if (i < n && pattern[j] != text[i]) {
                if (j != 0) j = lps[j - 1];
                else i++;
            }
        }
        return matches;
    }

    static vector<int> rabinKarpSearch(const string& text, const string& pattern) {
        vector<int> matches;
        const int prime = 101;
        const int d = 256;
        int m = pattern.length(), n = text.length();
        int p = 0, t = 0, h = 1;
        
        for (int i = 0; i < m - 1; i++) h = (h * d) % prime;
        for (int i = 0; i < m; i++) {
            p = (d * p + pattern[i]) % prime;
            t = (d * t + text[i]) % prime;
        }
        
        for (int i = 0; i <= n - m; i++) {
            if (p == t) {
                bool match = true;
                for (int j = 0; j < m; j++) {
                    if (text[i + j] != pattern[j]) {
                        match = false;
                        break;
                    }
                }
                if (match) matches.push_back(i);
            }
            if (i < n - m) {
                t = (d * (t - text[i] * h) + text[i + m]) % prime;
                if (t < 0) t += prime;
            }
        }
        return matches;
    }

    // Add new graph algorithm implementations
    static vector<int> depthFirstSearch(const Graph& graph, int start) {
        vector<int> result;
        vector<bool> visited(graph.V, false);
        stack<int> s;
        s.push(start);

        while (!s.empty()) {
            int v = s.top();
            s.pop();
            if (!visited[v]) {
                visited[v] = true;
                result.push_back(v);
                for (const auto& [u, _] : graph.adjacencyList[v]) {
                    if (!visited[u]) s.push(u);
                }
            }
        }
        return result;
    }

    static vector<int> breadthFirstSearch(const Graph& graph, int start) {
        vector<int> result;
        vector<bool> visited(graph.V, false);
        queue<int> q;
        q.push(start);
        visited[start] = true;

        while (!q.empty()) {
            int v = q.front();
            q.pop();
            result.push_back(v);
            for (const auto& [u, _] : graph.adjacencyList[v]) {
                if (!visited[u]) {
                    visited[u] = true;
                    q.push(u);
                }
            }
        }
        return result;
    }

    // Add Ford-Fulkerson implementation
    static int fordFulkerson(const Graph& graph, int source, int sink) {
        vector<vector<int>> residual(graph.V, vector<int>(graph.V, 0));
        for (const auto& edge : graph.edges) {
            residual[edge.source][edge.dest] = edge.weight;
        }

        vector<int> parent(graph.V, -1);
        auto bfs = [&](int s, int t) -> bool {
            fill(parent.begin(), parent.end(), -1);
            vector<bool> visited(graph.V, false);
            queue<int> q;
            q.push(s);
            visited[s] = true;

            while (!q.empty()) {
                int u = q.front();
                q.pop();
                for (int v = 0; v < graph.V; v++) {
                    if (!visited[v] && residual[u][v] > 0) {
                        q.push(v);
                        parent[v] = u;
                        visited[v] = true;
                        if (v == t) return true;
                    }
                }
            }
            return false;
        };

        int max_flow = 0;
        while (bfs(source, sink)) {
            int path_flow = numeric_limits<int>::max();
            for (int v = sink; v != source; v = parent[v]) {
                int u = parent[v];
                path_flow = min(path_flow, residual[u][v]);
            }
            for (int v = sink; v != source; v = parent[v]) {
                int u = parent[v];
                residual[u][v] -= path_flow;
                residual[v][u] += path_flow;
            }
            max_flow += path_flow;
        }
        return max_flow;
    }

    // Add Edmonds-Karp implementation
    static int edmondsKarp(const Graph& graph, int source, int sink) {
        vector<vector<int>> residual(graph.V, vector<int>(graph.V, 0));
        for (const auto& edge : graph.edges) {
            residual[edge.source][edge.dest] = edge.weight;
        }

        vector<int> parent(graph.V, -1);
        auto bfs = [&](int s, int t) -> bool {
            fill(parent.begin(), parent.end(), -1);
            vector<bool> visited(graph.V, false);
            queue<int> q;
            q.push(s);
            visited[s] = true;

            while (!q.empty()) {
                int u = q.front();
                q.pop();
                for (int v = 0; v < graph.V; v++) {
                    if (!visited[v] && residual[u][v] > 0) {
                        q.push(v);
                        parent[v] = u;
                        visited[v] = true;
                        if (v == t) return true;
                    }
                }
            }
            return false;
        };

        int max_flow = 0;
        while (bfs(source, sink)) {
            int path_flow = numeric_limits<int>::max();
            for (int v = sink; v != source; v = parent[v]) {
                int u = parent[v];
                path_flow = min(path_flow, residual[u][v]);
            }
            for (int v = sink; v != source; v = parent[v]) {
                int u = parent[v];
                residual[u][v] -= path_flow;
                residual[v][u] += path_flow;
            }
            max_flow += path_flow;
        }
        return max_flow;
    }

    // Add benchmark functionality
    template<typename Func>
    AlgorithmResult Benchmark(const std::string& name, Func algorithm) {
        AlgorithmResult result;
        result.algorithm_name = name;
        result.success = true;

        try {
            std::vector<double> times;
            for (int i = 0; i < Config::kBenchmarkIterations; ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                algorithm();
                auto end = std::chrono::high_resolution_clock::now();
                times.push_back(std::chrono::duration<double, std::milli>(
                    end - start).count());
            }

            // Calculate average time
            result.execution_time = std::accumulate(times.begin(), times.end(), 0.0) / 
                                  Config::kBenchmarkIterations;
            
            result.complexity = complexityMap[name].timeComplexity;
            Logger::LogPerformance(name, result.execution_time, result.complexity);
        } catch (const std::exception& e) {
            result.success = false;
            result.error_message = e.what();
            Logger::LogError(name + " failed: " + result.error_message);
        }

        return result;
    }

    // Add benchmark statistics structure
    struct BenchmarkStats {
        double min_time;
        double max_time;
        double avg_time;
        double std_deviation;
        size_t memory_used;
        vector<double> individual_times;
    };

    // Add memory tracking
    static size_t getCurrentMemoryUsage() {
        #ifdef _WIN32
            PROCESS_MEMORY_COUNTERS_EX pmc;
            if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
                return pmc.WorkingSetSize;
            }
        #elif defined(__APPLE__) && defined(__MACH__)
            struct mach_task_basic_info info;
            mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
            if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &infoCount) == KERN_SUCCESS) {
                return info.resident_size;
            }
        #elif defined(__linux__)
            long rss = 0L;
            FILE* fp = NULL;
            if ((fp = fopen("/proc/self/statm", "r")) != NULL) {
                if (fscanf(fp, "%*s%ld", &rss) == 1) {
                    fclose(fp);
                    return rss * sysconf(_SC_PAGESIZE);
                }
                fclose(fp);
            }
        #endif
        return 0;  // Return 0 if memory info is unavailable
    }

    // Enhanced benchmark functionality
    template<typename Func>
    static BenchmarkStats DetailedBenchmark(Func&& algorithm, int iterations) {
        BenchmarkStats stats{
            numeric_limits<double>::max(),  // min_time
            0.0,                           // max_time
            0.0,                           // avg_time
            0.0,                           // std_deviation
            0,                             // memory_used
            vector<double>()               // individual_times
        };

        size_t initial_memory = getCurrentMemoryUsage();
        
        // Warm-up run
        algorithm();

        // Actual benchmark runs
        for (int i = 0; i < iterations; ++i) {
            auto start = chrono::high_resolution_clock::now();
            algorithm();
            auto end = chrono::high_resolution_clock::now();
            
            double time = chrono::duration<double, milli>(end - start).count();
            stats.individual_times.push_back(time);
            stats.min_time = min(stats.min_time, time);
            stats.max_time = max(stats.max_time, time);
        }

        // Calculate statistics
        stats.avg_time = accumulate(stats.individual_times.begin(), 
                                  stats.individual_times.end(), 0.0) / iterations;
        
        double variance = 0.0;
        for (double time : stats.individual_times) {
            variance += pow(time - stats.avg_time, 2);
        }
        stats.std_deviation = sqrt(variance / iterations);
        
        stats.memory_used = getCurrentMemoryUsage() - initial_memory;
        
        return stats;
    }

public:
    // Add new public benchmark method with detailed statistics
    template<typename Func>
    static AlgorithmResult BenchmarkWithStats(
        const string& name,
        Func algorithm,
        int iterations = Config::kBenchmarkIterations,
        bool verbose = false
    ) {
        AlgorithmResult result;
        result.algorithm_name = name;
        result.success = true;

        try {
            auto stats = DetailedBenchmark(algorithm, iterations);
            result.execution_time = stats.avg_time;

            if (verbose) {
                cout << "\nDetailed Statistics for " << name << ":\n"
                     << "Min Time: " << stats.min_time << "ms\n"
                     << "Max Time: " << stats.max_time << "ms\n"
                     << "Avg Time: " << stats.avg_time << "ms\n"
                     << "Std Dev: " << stats.std_deviation << "ms\n"
                     << "Memory Used: " << (stats.memory_used / 1024.0) << "KB\n"
                     << "Distribution:\n";

                // Generate histogram of execution times
                map<int, int> histogram;
                for (double time : stats.individual_times) {
                    histogram[static_cast<int>(time * 10)]++;
                }

                for (const auto& [bucket, count] : histogram) {
                    cout << fixed << setprecision(1) 
                         << (bucket / 10.0) << "ms: "
                         << string(count, '*') << "\n";
                }
            }

            Logger::LogPerformance(name, stats.avg_time, "Custom Algorithm");
        } catch (const exception& e) {
            result.success = false;
            result.error_message = e.what();
            Logger::LogError(name + " failed: " + result.error_message);
        }

        return result;
    }

    // Add public benchmark method for users
    template<typename Func>
    static AlgorithmResult BenchmarkAlgorithm(
        const std::string& name,
        Func algorithm,
        int iterations = Config::kBenchmarkIterations
    ) {
        AlgorithmResult result;
        result.algorithm_name = name;
        result.success = true;

        try {
            std::vector<double> times;
            for (int i = 0; i < iterations; ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                algorithm();
                auto end = std::chrono::high_resolution_clock::now();
                times.push_back(std::chrono::duration<double, std::milli>(
                    end - start).count());
            }

            result.execution_time = std::accumulate(times.begin(), times.end(), 0.0) / iterations;
            Logger::LogPerformance(name, result.execution_time, "Custom Algorithm");
        } catch (const std::exception& e) {
            result.success = false;
            result.error_message = e.what();
            Logger::LogError(name + " failed: " + result.error_message);
        }

        return result;
    }

    void runPerformanceTest() {
        try {
            int size;
            cout << "Enter the number of elements to generate: ";
            if (!(cin >> size)) {
                throw runtime_error("Invalid input for size");
            }

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
        } catch (const exception& e) {
            cout << "Error in performance test: " << e.what() << "\n";
        }
    }

    void runGraphAlgorithms() {
        try {
            int V, E;
            cout << "Enter number of vertices (1-1000): ";
            if (!(cin >> V) || V <= 0 || V > 1000) {
                throw runtime_error("Invalid number of vertices");
            }

            cout << "Enter number of edges (0-" << V * (V-1) << "): ";
            if (!(cin >> E) || E < 0 || E > V * (V-1)) {
                throw runtime_error("Invalid number of edges");
            }

            Graph graph(V, E);

            cout << "Enter edge information (source destination weight):\n";
            for (int i = 0; i < E; i++) {
                int src, dest, weight;
                if (!(cin >> src >> dest >> weight)) {
                    throw runtime_error("Invalid edge input");
                }
                if (src < 0 || src >= V || dest < 0 || dest >= V) {
                    throw runtime_error("Invalid vertex in edge");
                }
                if (weight < 0 || weight > INT_MAX/2) {  // Prevent overflow in path calculations
                    throw runtime_error("Invalid edge weight");
                }
                graph.addEdge(src, dest, weight);
            }

            // Add algorithm selection menu
            cout << "\nSelect graph algorithms to run:\n";
            cout << "1. Shortest Path Algorithms (Dijkstra, Bellman-Ford, Floyd-Warshall)\n";
            cout << "2. Graph Traversal (DFS, BFS)\n";
            cout << "3. Minimum Spanning Tree (Prim's, Kruskal's)\n";
            cout << "4. Flow Networks (Ford-Fulkerson, Edmonds-Karp)\n";
            cout << "5. All Algorithms\n";
            
            int algorithmChoice;
            cout << "Enter your choice (1-5): ";
            if (!(cin >> algorithmChoice)) {
                throw runtime_error("Invalid algorithm choice");
            }

            int source;
            cout << "Enter source vertex: ";
            if (!(cin >> source) || source < 0 || source >= V) {
                throw runtime_error("Invalid source vertex");
            }

            // Vector to store performance results
            vector<AlgorithmResult> results;

            // Run selected algorithms based on user choice
            if (algorithmChoice == 1 || algorithmChoice == 5) {
                // Shortest Path Algorithms
                cout << "\n--- Shortest Path Algorithms ---\n";
                
                // Test Dijkstra's Algorithm
                auto result = BenchmarkWithStats("Dijkstra",
                    [&]() { dijkstra(graph, source); }, 5, true);
                results.push_back(result);

                // Test Bellman-Ford Algorithm
                auto bellmanFordResult = BenchmarkWithStats("Bellman-Ford",
                    [&]() { bellmanFord(graph, source); }, 5, true);
                results.push_back(bellmanFordResult);

                // Add Floyd-Warshall testing after Bellman-Ford
                auto floydWarshallResult = BenchmarkWithStats("Floyd-Warshall",
                    [&]() { floydWarshall(graph); }, 5, true);
                results.push_back(floydWarshallResult);
            }

            if (algorithmChoice == 2 || algorithmChoice == 5) {
                // Graph Traversal
                cout << "\n--- Graph Traversal Algorithms ---\n";
                
                // Test DFS
                auto dfsResult = BenchmarkWithStats("DFS",
                    [&]() { depthFirstSearch(graph, source); }, 5, true);
                results.push_back(dfsResult);
                
                // Test BFS
                auto bfsResult = BenchmarkWithStats("BFS",
                    [&]() { breadthFirstSearch(graph, source); }, 5, true);
                results.push_back(bfsResult);

                // Display traversal results
                vector<int> dfsPath = depthFirstSearch(graph, source);
                vector<int> bfsPath = breadthFirstSearch(graph, source);

                cout << "\nDFS Path from vertex " << source << ": ";
                for (int v : dfsPath) cout << v << " ";
                cout << "\n";

                cout << "BFS Path from vertex " << source << ": ";
                for (int v : bfsPath) cout << v << " ";
                cout << "\n";
            }

            if (algorithmChoice == 3 || algorithmChoice == 5) {
                // Minimum Spanning Tree
                cout << "\n--- Minimum Spanning Tree Algorithms ---\n";
                
                // Test Prim's MST
                auto primResult = BenchmarkWithStats("Prim's MST",
                    [&]() { primMST(graph); }, 5, true);
                results.push_back(primResult);

                vector<Edge> mst = primMST(graph);
                cout << "\nPrim's MST Edges:\n";
                int totalWeight = 0;
                for (const Edge& e : mst) {
                    cout << e.source << " -- " << e.dest << " (weight: " << e.weight << ")\n";
                    totalWeight += e.weight;
                }
                cout << "Total MST Weight: " << totalWeight << "\n";

                // Test Kruskal's MST
                auto kruskalResult = BenchmarkWithStats("Kruskal's MST",
                    [&]() { kruskalMST(graph); }, 5, true);
                results.push_back(kruskalResult);
            }

            if (algorithmChoice == 4 || algorithmChoice == 5) {
                // Flow Networks
                cout << "\n--- Flow Network Algorithms ---\n";

                int sink;
                cout << "Enter sink vertex for Ford-Fulkerson and Edmonds-Karp: ";
                if (!(cin >> sink) || sink < 0 || sink >= graph.V) {
                    throw runtime_error("Invalid sink vertex");
                }

                // Test Ford-Fulkerson
                auto fordFulkersonResult = BenchmarkWithStats("Ford-Fulkerson",
                    [&]() { fordFulkerson(graph, source, sink); }, 5, true);
                results.push_back(fordFulkersonResult);

                // Test Edmonds-Karp
                auto edmondsKarpResult = BenchmarkWithStats("Edmonds-Karp",
                    [&]() { edmondsKarp(graph, source, sink); }, 5, true);
                results.push_back(edmondsKarpResult);
            }

            // Display performance comparison
            cout << "\n--- Performance Comparison ---\n";
            PerformanceVisualizer::DisplayBarChart(results);

            // ...rest of existing code...

        } catch (const exception& e) {
            cout << "Error in graph algorithms: " << e.what() << "\n";
        }
    }

    // Add new function to run DP and String Matching algorithms
    void runAdvancedAlgorithms() {
        try {
            cout << "\n--- Dynamic Programming Algorithms ---\n";
            
            // Knapsack Problem
            cout << "Enter number of items for Knapsack: ";
            int n;
            cin >> n;
            vector<int> values(n), weights(n);
            cout << "Enter values and weights:\n";
            for (int i = 0; i < n; i++) {
                cout << "Item " << i + 1 << " - Value: ";
                cin >> values[i];
                cout << "Item " << i + 1 << " - Weight: ";
                cin >> weights[i];
            }
            int capacity;
            cout << "Enter knapsack capacity: ";
            cin >> capacity;

            auto [maxValue, selected] = knapsack(values, weights, capacity);
            cout << "Maximum value: " << maxValue << "\nSelected items: ";
            for (int i : selected) cout << i << " ";
            cout << "\n";

            // Longest Increasing Subsequence
            cout << "\nLongest Increasing Subsequence of the values:\n";
            auto lis = longestIncreasingSubsequence(values);
            cout << "Length: " << lis.size() << "\nSequence: ";
            for (int x : lis) cout << x << " ";
            cout << "\n";

            // Matrix Chain Multiplication
            cout << "\nEnter number of matrices: ";
            int m;
            cin >> m;
            vector<int> dimensions(m + 1);
            cout << "Enter matrix dimensions (p0, p1, ..., pm):\n";
            for (int i = 0; i <= m; i++) {
                cout << "p" << i << ": ";
                cin >> dimensions[i];
            }

            auto [minOps, order] = matrixChainMultiplication(dimensions);
            cout << "Minimum operations: " << minOps << "\nMultiplication order: ";
            for (int x : order) cout << x << " ";
            cout << "\n";

            // String Matching
            cout << "\n--- String Matching Algorithms ---\n";
            string text, pattern;
            cout << "Enter text: ";
            cin.ignore(); // Clear the newline character from the input buffer
            getline(cin, text);
            cout << "Enter pattern: ";
            getline(cin, pattern);

            auto kmpMatches = KMPSearch(text, pattern);
            cout << "\nKMP matches found at positions: ";
            for (int pos : kmpMatches) cout << pos << " ";

            auto rkMatches = rabinKarpSearch(text, pattern);
            cout << "\nRabin-Karp matches found at positions: ";
            for (int pos : rkMatches) cout << pos << " ";
            cout << "\n";

        } catch (const exception& e) {
            cout << "Error in advanced algorithms: " << e.what() << "\n";
        }
    }

    // Update the public interface with new features
    void RunPerformanceTest() {
        try {
            int size;
            cout << "Enter the number of elements to generate: ";
            if (!(cin >> size)) {
                throw runtime_error("Invalid input for size");
            }

            generateRandomData(size);

            // Define sorting algorithms collection
            vector<pair<string, function<void(vector<int>&)>>> sortingAlgorithms = {
                {"Bubble Sort", [this](vector<int>& arr) { bubbleSort(arr); }},
                {"Selection Sort", [this](vector<int>& arr) { selectionSort(arr); }},
                {"Quick Sort", [this](vector<int>& arr) { quickSort(arr, 0, arr.size() - 1); }},
                {"Merge Sort", [this](vector<int>& arr) { mergeSort(arr, 0, arr.size() - 1); }},
                {"Heap Sort", [this](vector<int>& arr) { heapSort(arr); }},
                {"Count Sort", [this](vector<int>& arr) { countSort(arr); }},
                {"Radix Sort", [this](vector<int>& arr) { radixSort(arr); }},
                {"Bucket Sort", [this](vector<int>& arr) { bucketSort(arr); }},
                {"C++ Standard Sort", [](vector<int>& arr) { sort(arr.begin(), arr.end()); }}
            };

            // Sort the array for binary search
            sort(workingData.begin(), workingData.end());

            // Get search target
            int searchTarget;
            cout << "\nEnter a value to search: ";
            cin >> searchTarget;

            // Define searching algorithms collection
            vector<pair<string, function<int(const vector<int>&, int)>>> searchAlgorithms = {
                {"Linear Search", linearSearch},
                {"Binary Search", [this](const vector<int>& arr, int target) { 
                    return binarySearch(const_cast<vector<int>&>(arr), target); 
                }}
            };

            // Vector to store all algorithm results
            std::vector<AlgorithmResult> results;
            
            // Benchmark sorting algorithms
            for (const auto& algo : sortingAlgorithms) {
                results.push_back(Benchmark(algo.first, 
                    [&]() { algo.second(workingData); }));
            }

            // Benchmark searching algorithms
            for (const auto& algo : searchAlgorithms) {
                results.push_back(Benchmark(algo.first,
                    [&]() { algo.second(workingData, searchTarget); }));
            }

            // Rest of the implementation
            // ...existing code...
        } catch (const std::exception& e) {
            Logger::LogError("Performance test failed: " + string(e.what()));
            throw;
        }
    }

    // Add configuration methods
    static void SetBenchmarkIterations(int iterations) {
        if (iterations > 0) {
            const_cast<int&>(Config::kBenchmarkIterations) = iterations;
        }
    }

    // Add new public method for custom benchmarking from menu
    static void runCustomBenchmark() {
        try {
            cout << "\n=== Enhanced Custom Algorithm Benchmark ===\n";
            cout << "Select benchmark type:\n";
            cout << "1. Vector operations\n";
            cout << "2. String operations\n";
            cout << "3. Custom data operation\n";
            cout << "4. Algorithm comparison\n";
            cout << "5. Memory usage analysis\n";
            
            int choice;
            cout << "Enter choice: ";
            if (!(cin >> choice)) {
                throw runtime_error("Invalid input");
            }

            switch (choice) {
                case 1: {
                    int size;
                    cout << "Enter vector size: ";
                    if (!(cin >> size) || size <= 0) {
                        throw runtime_error("Invalid size");
                    }
                    
                    vector<int> data(size);
                    generate(data.begin(), data.end(), rand);

                    auto result = BenchmarkAlgorithm(
                        "Vector Sort",
                        [&]() { sort(data.begin(), data.end()); }
                    );

                    cout << "\nBenchmark Results:\n";
                    cout << "Algorithm: " << result.algorithm_name << "\n";
                    cout << "Time: " << result.execution_time << "ms\n";
                    cout << "Status: " << (result.success ? "Success" : "Failed") << "\n";
                    if (!result.success) {
                        cout << "Error: " << result.error_message << "\n";
                    }
                    break;
                }
                case 2: {
                    string text;
                    cout << "Enter text to process: ";
                    cin.ignore();
                    getline(cin, text);

                    auto result = BenchmarkAlgorithm(
                        "String Reverse",
                        [&]() { string copy = text; reverse(copy.begin(), copy.end()); }
                    );

                    cout << "\nBenchmark Results:\n";
                    cout << "Algorithm: " << result.algorithm_name << "\n";
                    cout << "Time: " << result.execution_time << "ms\n";
                    cout << "Status: " << (result.success ? "Success" : "Failed") << "\n";
                    break;
                }
                case 3: {
                    cout << "Enter number of iterations: ";
                    int iterations;
                    if (!(cin >> iterations) || iterations <= 0) {
                        throw runtime_error("Invalid iterations count");
                    }

                    auto result = BenchmarkAlgorithm(
                        "Custom Operation",
                        []() { 
                            // Example custom operation
                            volatile int sum = 0;
                            for(int i = 0; i < 1000000; i++) sum += i;
                        },
                        iterations
                    );

                    cout << "\nBenchmark Results:\n";
                    cout << "Algorithm: " << result.algorithm_name << "\n";
                    cout << "Time: " << result.execution_time << "ms\n";
                    cout << "Status: " << (result.success ? "Success" : "Failed") << "\n";
                    cout << "Iterations: " << iterations << "\n";
                    break;
                }
                case 4: {
                    // Algorithm comparison
                    cout << "Enter vector size for comparison: ";
                    int size;
                    cin >> size;
                    vector<int> data(size);
                    generate(data.begin(), data.end(), rand);

                    vector<AlgorithmResult> results;
                    
                    // Compare different sorting algorithms
                    results.push_back(BenchmarkWithStats("STL Sort",
                        [&]() { sort(data.begin(), data.end()); }, 10, true));
                    
                    results.push_back(BenchmarkWithStats("Quick Sort",
                        [&]() { 
                            vector<int> temp = data;
                            quickSort(temp, 0, temp.size() - 1); 
                        }, 10, true));

                    PerformanceVisualizer::DisplayBarChart(results);
                    break;
                }

                case 5: {
                    // Memory usage analysis
                    cout << "Enter data structure size: ";
                    int size;
                    cin >> size;

                    BenchmarkWithStats("Vector Memory",
                        [size]() {
                            vector<int> vec(size);
                            generate(vec.begin(), vec.end(), rand);
                            sort(vec.begin(), vec.end());
                        }, 5, true);
                    break;
                }
                default:
                    throw runtime_error("Invalid choice");
            }
        } catch (const exception& e) {
            cout << "Benchmark error: " << e.what() << "\n";
        }
    }
};

int main() {
    try {
        AlgorithmPerformanceTester tester;
        
        int choice;
        do {
            cout << "\n=== Algorithm Performance Testing Framework ===\n";
            cout << "1. Test Sorting and Searching Algorithms\n";
            cout << "2. Test Graph Algorithms\n";
            cout << "3. Test Dynamic Programming and String Matching\n";
            cout << "4. Run Custom Benchmark\n";  // New option
            cout << "5. Exit\n";
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
                    tester.runAdvancedAlgorithms();
                    break;
                case 4:
                    AlgorithmPerformanceTester::runCustomBenchmark();
                    break;
                case 5:
                    cout << "Exiting...\n";
                    break;
                default:
                    cout << "Invalid choice!\n";
            }
        } while (choice != 5);
    } catch (const exception& e) {
        Logger::LogError("Fatal error in main: " + std::string(e.what()));
        cout << "Fatal error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}