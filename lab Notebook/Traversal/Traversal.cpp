#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;

void BFS(const vector<vector<int>>& graph, int start) {
    vector<bool> visited(graph.size(), false);
    queue<int> q;
    q.push(start);
    visited[start] = true;

    while (!q.empty()) {
        int node = q.front();
        q.pop();
        cout << node << " ";

        for (int neighbor : graph[node]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}

void DFS(const vector<vector<int>>& graph, int start) {
    vector<bool> visited(graph.size(), false);
    stack<int> s;
    s.push(start);

    while (!s.empty()) {
        int node = s.top();
        s.pop();

        if (!visited[node]) {
            visited[node] = true;
            cout << node << " ";
        }

        for (int neighbor : graph[node]) {
            if (!visited[neighbor]) {
                s.push(neighbor);
            }
        }
    }
}

bool DLS(const vector<vector<int>>& graph, int node, int target, int depth, vector<bool>& visited) {
    if (node == target) return true;
    if (depth <= 0) return false;

    visited[node] = true;
    for (int neighbor : graph[node]) {
        if (!visited[neighbor]) {
            if (DLS(graph, neighbor, target, depth - 1, visited)) return true;
        }
    }
    visited[node] = false;
    return false;
}

void IDS(const vector<vector<int>>& graph, int start, int target) {
    for (int depth = 0; depth < graph.size(); ++depth) {
        vector<bool> visited(graph.size(), false);
        if (DLS(graph, start, target, depth, visited)) {
            cout << "Target " << target << " found at depth " << depth << endl;
            return;
        }
    }
    cout << "Target " << target << " not found" << endl;
}

int main() {
    int n, e;
    cout << "Enter number of nodes and edges: ";
    cin >> n >> e;

    vector<vector<int>> graph(n);
    cout << "Enter edges (u v):" << endl;
    for (int i = 0; i < e; ++i) {
        int u, v;
        cin >> u >> v;
        graph[u].push_back(v);
        graph[v].push_back(u); // Assuming undirected graph
    }

    int start;
    cout << "Enter start node: ";
    cin >> start;

    auto start_time = high_resolution_clock::now();
    cout << "BFS Traversal: ";
    BFS(graph, start);
    auto end_time = high_resolution_clock::now();
    auto bfs_duration = duration_cast<microseconds>(end_time - start_time).count();
    cout << "\nTime taken by BFS: " << bfs_duration << " microseconds" << endl;

    start_time = high_resolution_clock::now();
    cout << "DFS Traversal: ";
    DFS(graph, start);
    end_time = high_resolution_clock::now();
    auto dfs_duration = duration_cast<microseconds>(end_time - start_time).count();
    cout << "\nTime taken by DFS: " << dfs_duration << " microseconds" << endl;

    int target;
    cout << "Enter target node for IDS: ";
    cin >> target;

    start_time = high_resolution_clock::now();
    cout << "IDS Traversal: ";
    IDS(graph, start, target);
    end_time = high_resolution_clock::now();
    auto ids_duration = duration_cast<microseconds>(end_time - start_time).count();
    cout << "Time taken by IDS: " << ids_duration << " microseconds" << endl;

    cout << "\nTraversal Statistics:\n";
    cout << left << setw(10) << "Algorithm" << setw(20) << "Time (microseconds)" << setw(20) << "Space Complexity" << endl;
    cout << left << setw(10) << "BFS" << setw(20) << bfs_duration << setw(20) << "O(V + E)" << endl;
    cout << left << setw(10) << "DFS" << setw(20) << dfs_duration << setw(20) << "O(V + E)" << endl;
    cout << left << setw(10) << "IDS" << setw(20) << ids_duration << setw(20) << "O(V + E)" << endl;

    return 0;
}
