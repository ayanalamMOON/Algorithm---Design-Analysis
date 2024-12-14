#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
using namespace std;

// Structure for graph edges
struct Edge {
    int src, dest, weight;
};

// Structure for disjoint sets
struct DisjointSet {
    vector<int> parent, rank;
    DisjointSet(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        for (int i = 0; i < n; i++)
            parent[i] = i;
    }
    
    int find(int u) {
        if (parent[u] != u)
            parent[u] = find(parent[u]);
        return parent[u];
    }
    
    void unite(int u, int v) {
        int pu = find(u), pv = find(v);
        if (rank[pu] < rank[pv])
            parent[pu] = pv;
        else if (rank[pu] > rank[pv])
            parent[pv] = pu;
        else {
            parent[pv] = pu;
            rank[pu]++;
        }
    }
};

// Kruskal's Algorithm implementation
vector<Edge> kruskalMST(vector<Edge>& edges, int V) {
    vector<Edge> result;
    DisjointSet ds(V);
    
    // Sort edges by weight
    sort(edges.begin(), edges.end(), 
         [](Edge a, Edge b) { return a.weight < b.weight; });
    
    for (Edge e : edges) {
        int u = ds.find(e.src);
        int v = ds.find(e.dest);
        
        if (u != v) {
            result.push_back(e);
            ds.unite(u, v);
        }
    }
    return result;
}

int main() {
    // Get input from user
    int V, E;
    cout << "Enter number of vertices: ";
    cin >> V;
    cout << "Enter number of edges: ";
    cin >> E;
    
    vector<Edge> edges(E);
    cout << "\nEnter edges (source destination weight):\n";
    for (int i = 0; i < E; i++) {
        cin >> edges[i].src >> edges[i].dest >> edges[i].weight;
    }
    
    // Time measurement start
    auto start = chrono::high_resolution_clock::now();
    
    // Perform Kruskal's Algorithm
    vector<Edge> mst = kruskalMST(edges, V);
    
    // Time measurement end
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    
    // Print results
    cout << "\nMinimum Spanning Tree edges:\n";
    int totalWeight = 0;
    for (Edge e : mst) {
        cout << e.src << " -- " << e.dest << " : " << e.weight << endl;
        totalWeight += e.weight;
    }
    cout << "\nTotal MST weight: " << totalWeight << endl;
    cout << "Time taken: " << duration.count() << " microseconds" << endl;
    
    // Print complexity analysis
    cout << "\nComplexity Analysis:" << endl;
    cout << "Time Complexity: O(E log E)" << endl;
    cout << "Space Complexity: O(V + E)" << endl;
    cout << "Where V is the number of vertices and E is the number of edges" << endl;
    
    return 0;
}
