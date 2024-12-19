#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <chrono>
using namespace std;

class Graph {
    int V;
    vector<vector<pair<int, int>>> adj;

public:
    Graph(int vertices) : V(vertices) {
        adj.resize(V);
    }

    void addEdge(int u, int v, int weight) {
        adj[u].push_back({v, weight});
        adj[v].push_back({u, weight}); // For undirected graph
    }

    vector<int> dijkstra(int src) {
        vector<int> dist(V, numeric_limits<int>::max());
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;

        dist[src] = 0;
        pq.push({0, src});

        while (!pq.empty()) {
            int u = pq.top().second;
            pq.pop();

            for (auto& neighbor : adj[u]) {
                int v = neighbor.first;
                int weight = neighbor.second;

                if (dist[v] > dist[u] + weight) {
                    dist[v] = dist[u] + weight;
                    pq.push({dist[v], v});
                }
            }
        }
        return dist;
    }
};

int main() {
    int V, E;
    cout << "Enter number of vertices: ";
    cin >> V;
    cout << "Enter number of edges: ";
    cin >> E;

    Graph g(V);

    cout << "Enter edges (u v weight):\n";
    for (int i = 0; i < E; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        g.addEdge(u, v, w);
    }

    int source;
    cout << "Enter source vertex: ";
    cin >> source;

    // Start timing
    auto start = chrono::high_resolution_clock::now();
    
    vector<int> distances = g.dijkstra(source);
    
    // End timing
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    // Print results
    cout << "\nShortest distances from vertex " << source << ":\n";
    for (int i = 0; i < V; i++) {
        cout << "To vertex " << i << ": " << distances[i] << endl;
    }

    // Print complexity analysis
    cout << "\nComplexity Analysis:" << endl;
    cout << "Time Complexity: O((V + E)logV)" << endl;
    cout << "Space Complexity: O(V)" << endl;
    cout << "Actual time taken: " << duration.count() << " microseconds" << endl;

    return 0;
}
