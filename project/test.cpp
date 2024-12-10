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
    vector<vector<int>> grid;  // For A* pathfinding

    Graph(int vertices, int edges) : V(vertices), E(edges) {
        adjacencyList.resize(V);
        grid.resize(V, vector<int>(V, 1)); // Initialize grid for A*
    }

    void addEdge(int src, int dest, int weight) {
        edges.push_back({src, dest, weight});
        adjacencyList[src].push_back({dest, weight});
        // For undirected graph algorithms (MST)
        adjacencyList[dest].push_back({src, weight});
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
        {"Rabin-Karp", {"O(n + m)", "O(1)"}}
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
            if (pattern[i] == pattern[len]) {
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

public:
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

            int source;
            cout << "Enter source vertex: ";
            if (!(cin >> source) || source < 0 || source >= V) {
                throw runtime_error("Invalid source vertex");
            }

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

            // Add new algorithm tests
            cout << "\n--- Additional Graph Algorithms ---\n";

            // Test Prim's MST
            start = chrono::high_resolution_clock::now();
            vector<Edge> primResult = primMST(graph);
            end = chrono::high_resolution_clock::now();
            double primTime = chrono::duration<double, milli>(end - start).count();

            // Test Kruskal's MST
            start = chrono::high_resolution_clock::now();
            vector<Edge> kruskalResult = kruskalMST(graph);
            end = chrono::high_resolution_clock::now();
            double kruskalTime = chrono::duration<double, milli>(end - start).count();

            // Test A* pathfinding
            int goal;
            cout << "Enter goal vertex for A* pathfinding: ";
            if (!(cin >> goal) || goal < 0 || goal >= graph.V) {
                throw runtime_error("Invalid goal vertex");
            }

            start = chrono::high_resolution_clock::now();
            vector<int> astarResult = astar(graph, source, goal);
            end = chrono::high_resolution_clock::now();
            double astarTime = chrono::duration<double, milli>(end - start).count();

            // Test Topological Sort
            start = chrono::high_resolution_clock::now();
            vector<int> topoResult = topologicalSort(graph);
            end = chrono::high_resolution_clock::now();
            double topoTime = chrono::duration<double, milli>(end - start).count();

            // Print results
            cout << "\nMinimum Spanning Tree Results:\n";
            cout << "Prim's MST weight: " << accumulate(primResult.begin(), primResult.end(), 0,
                [](int sum, const Edge& e) { return sum + e.weight; }) << "\n";
            cout << "Kruskal's MST weight: " << accumulate(kruskalResult.begin(), kruskalResult.end(), 0,
                [](int sum, const Edge& e) { return sum + e.weight; }) << "\n";

            cout << "\nA* Path from " << source << " to " << goal << ": ";
            for (int v : astarResult) cout << v << " ";
            cout << "\n";

            cout << "\nTopological Sort: ";
            for (int v : topoResult) cout << v << " ";
            cout << "\n";

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
};

int main() {
    try {
        AlgorithmPerformanceTester tester;
        
        int choice;
        do {
            cout << "\n1. Test Sorting and Searching Algorithms\n";
            cout << "2. Test Graph Algorithms\n";
            cout << "3. Test Dynamic Programming and String Matching\n";
            cout << "4. Exit\n";
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
                    cout << "Exiting...\n";
                    break;
                default:
                    cout << "Invalid choice!\n";
            }
        } while (choice != 4);
    } catch (const exception& e) {
        cout << "Fatal error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}