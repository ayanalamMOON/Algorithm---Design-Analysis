#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <queue>
#include <limits>
#include <algorithm>
#include <unordered_map>
#include <set>
#include <cmath>

using namespace std;
using namespace chrono;

// Dijkstra's Algorithm
void dijkstra(const vector<vector<int>>& graph, int src) {
    int n = graph.size();
    vector<int> dist(n, numeric_limits<int>::max());
    dist[src] = 0;
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
    pq.push({0, src});

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();

        for (int v = 0; v < n; ++v) {
            if (graph[u][v] && dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
                pq.push({dist[v], v});
            }
        }
    }
}

// Bellman-Ford Algorithm
void bellmanFord(const vector<vector<int>>& graph, int src) {
    int n = graph.size();
    vector<int> dist(n, numeric_limits<int>::max());
    dist[src] = 0;

    for (int i = 1; i < n; ++i) {
        for (int u = 0; u < n; ++u) {
            for (int v = 0; v < n; ++v) {
                if (graph[u][v] && dist[u] != numeric_limits<int>::max() && dist[u] + graph[u][v] < dist[v]) {
                    dist[v] = dist[u] + graph[u][v];
                }
            }
        }
    }
}

// Floyd-Warshall Algorithm
void floydWarshall(const vector<vector<int>>& graph) {
    int n = graph.size();
    vector<vector<int>> dist = graph;

    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (dist[i][k] != numeric_limits<int>::max() && dist[k][j] != numeric_limits<int>::max() && dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }
}

// Johnson's Algorithm
void johnson(const vector<vector<int>>& originalGraph) {
    int n = originalGraph.size();
    vector<vector<int>> graph = originalGraph; // Create a local copy
    vector<vector<int>> newGraph = graph;
    vector<int> h(n + 1, 0);

    // Add a new vertex and connect it to all other vertices with edge weight 0
    for (int i = 0; i < n; ++i) {
        newGraph.push_back(vector<int>(n + 1, 0));
        newGraph[i].push_back(0);
    }
    newGraph.push_back(vector<int>(n + 1, 0));

    // Run Bellman-Ford to find shortest path from new vertex to all other vertices
    bellmanFord(newGraph, n);

    // Reweight the edges using the local copy
    for (int u = 0; u < n; ++u) {
        for (int v = 0; v < n; ++v) {
            if (graph[u][v] != numeric_limits<int>::max()) {
                graph[u][v] += h[u] - h[v];
            }
        }
    }

    // Run Dijkstra's algorithm for each vertex using the reweighted graph
    for (int u = 0; u < n; ++u) {
        dijkstra(graph, u);
    }
}

// A* (A-star) Algorithm
int heuristic(int u, int v) {
    // Convert vertex indices to 2D coordinates
    int ux = u / 10, uy = u % 10;
    int vx = v / 10, vy = v % 10;
    
    // Manhattan distance
    int manhattan = abs(ux - vx) + abs(uy - vy);
    
    // Euclidean distance
    double euclidean = sqrt(pow(ux - vx, 2) + pow(uy - vy, 2));
    
    // Combine both heuristics (weighted combination)
    return static_cast<int>((manhattan + euclidean) * 0.5);
}

void aStar(const vector<vector<int>>& graph, int src, int dest) {
    int n = graph.size();
    vector<int> dist(n, numeric_limits<int>::max());
    dist[src] = 0;
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
    pq.push({0, src});

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();

        if (u == dest) {
            return;
        }

        for (int v = 0; v < n; ++v) {
            if (graph[u][v] && dist[u] + graph[u][v] + heuristic(v, dest) < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
                pq.push({dist[v] + heuristic(v, dest), v});
            }
        }
    }
}

// Bidirectional Search
void bidirectionalSearch(const vector<vector<int>>& graph, int src, int dest) {
    int n = graph.size();
    vector<bool> visitedFromSrc(n, false), visitedFromDest(n, false);
    queue<int> qSrc, qDest;
    qSrc.push(src);
    qDest.push(dest);
    visitedFromSrc[src] = true;
    visitedFromDest[dest] = true;

    while (!qSrc.empty() && !qDest.empty()) {
        int u = qSrc.front();
        qSrc.pop();
        for (int v = 0; v < n; ++v) {
            if (graph[u][v] && !visitedFromSrc[v]) {
                visitedFromSrc[v] = true;
                qSrc.push(v);
                if (visitedFromDest[v]) {
                    return;
                }
            }
        }

        int w = qDest.front();
        qDest.pop();
        for (int v = 0; v < n; ++v) {
            if (graph[w][v] && !visitedFromDest[v]) {
                visitedFromDest[v] = true;
                qDest.push(v);
                if (visitedFromSrc[v]) {
                    return;
                }
            }
        }
    }
}

void printStats(const string& algorithm, double timeTaken, const string& timeComplexity, const string& spaceComplexity) {
    cout << left << setw(20) << algorithm
         << setw(20) << timeTaken
         << setw(20) << timeComplexity
         << setw(20) << spaceComplexity << endl;
}

int main() {
    int n, src, dest;
    cout << "Enter the number of vertices: ";
    cin >> n;
    vector<vector<int>> graph(n, vector<int>(n));

    cout << "Enter the adjacency matrix:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cin >> graph[i][j];
        }
    }

    cout << "Enter the source vertex: ";
    cin >> src;
    cout << "Enter the destination vertex: ";
    cin >> dest;

    cout << left << setw(20) << "Algorithm"
         << setw(20) << "Time Taken (ms)"
         << setw(20) << "Time Complexity"
         << setw(20) << "Space Complexity" << endl;

    auto start = high_resolution_clock::now();
    dijkstra(graph, src);
    auto end = high_resolution_clock::now();
    double timeTaken = duration<double, milli>(end - start).count();
    printStats("Dijkstra", timeTaken, "O(V^2)", "O(V)");

    start = high_resolution_clock::now();
    bellmanFord(graph, src);
    end = high_resolution_clock::now();
    timeTaken = duration<double, milli>(end - start).count();
    printStats("Bellman-Ford", timeTaken, "O(VE)", "O(V)");

    start = high_resolution_clock::now();
    floydWarshall(graph);
    end = high_resolution_clock::now();
    timeTaken = duration<double, milli>(end - start).count();
    printStats("Floyd-Warshall", timeTaken, "O(V^3)", "O(V^2)");

    start = high_resolution_clock::now();
    johnson(graph);
    end = high_resolution_clock::now();
    timeTaken = duration<double, milli>(end - start).count();
    printStats("Johnson", timeTaken, "O(V^2 log V + VE)", "O(V^2)");

    start = high_resolution_clock::now();
    aStar(graph, src, dest);
    end = high_resolution_clock::now();
    timeTaken = duration<double, milli>(end - start).count();
    printStats("A*", timeTaken, "O(E)", "O(V)");

    start = high_resolution_clock::now();
    bidirectionalSearch(graph, src, dest);
    end = high_resolution_clock::now();
    timeTaken = duration<double, milli>(end - start).count();
    printStats("Bidirectional Search", timeTaken, "O(b^(d/2))", "O(b^(d/2))");

    return 0;
}
