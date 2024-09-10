#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <cassert>

using namespace std;

class Solution {
public:
    vector<vector<pair<int, int>>> adj;
    
    vector<long long> dijkstra(int n, int start, vector<int> &weights) {
        vector<long long> dist(n, numeric_limits<long long>::max());
        priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
        dist[start] = 0;
        pq.push({0, start});
        
        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if (d > dist[u]) continue;
            for (auto &[v, index] : adj[u]) {
                long long nextDist = d + (weights[index] == -1 ? 1 : weights[index]);
                if (nextDist < dist[v]) {
                    dist[v] = nextDist;
                    pq.push({nextDist, v});
                }
            }
        }
        return dist;
    }
    
    vector<vector<int>> modifiedGraphEdges(int n, vector<vector<int>>& edges, int source, int destination, int target) {
        adj = vector<vector<pair<int, int>>>(n);
        vector<int> weights(edges.size(), -1);
        vector<int> negEdgeIndexes;

        for (int i = 0; i < edges.size(); ++i) {
            int u = edges[i][0], v = edges[i][1], w = edges[i][2];
            adj[u].push_back({v, i});
            adj[v].push_back({u, i});
            if (w == -1) negEdgeIndexes.push_back(i);
            else weights[i] = w;
        }

        auto dist = dijkstra(n, source, weights);
        if (dist[destination] < target) return {};
        if (dist[destination] == target) return edges;

        long long low = 1, high = 2e9;
        while (low <= high) {
            long long mid = low + (high - low) / 2;
            for (int i : negEdgeIndexes) {
                weights[i] = mid;
            }
            dist = dijkstra(n, source, weights);
            if (dist[destination] == target) {
                for (int i : negEdgeIndexes) {
                    edges[i][2] = weights[i];
                }
                return edges;
            } else if (dist[destination] < target) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }

        return {};
    }
};

class SolutionTest {
public:
    void runTests() {
        Solution sol;
        
        // Test case 1
        assert(sol.modifiedGraphEdges(5, {{0,1,1},{1,2,-1},{2,3,1},{0,3,3},{0,4,2},{3,4,2}}, 0, 3, 5) 
            == vector<vector<int>>{{0,1,1},{1,2,2},{2,3,1},{0,3,3},{0,4,2},{3,4,2}});
        cout << "Test case 1 passed." << endl;
        
        // Test case 2
        assert(sol.modifiedGraphEdges(3, {{0,1,1},{1,2,1}}, 0, 2, 2) 
            == vector<vector<int>>{{0,1,1},{1,2,1}});
        cout << "Test case 2 passed." << endl;
        
        // Test case 3
        assert(sol.modifiedGraphEdges(3, {{0,1,1},{1,2,1}}, 0, 2, 1).empty());
        cout << "Test case 3 passed." << endl;
        
        // Test case 4
        assert(sol.modifiedGraphEdges(4, {{0,1,-1},{0,2,-1},{1,2,-1},{1,3,1}}, 0, 3, 4) 
            == vector<vector<int>>{{0,1,2},{0,2,1},{1,2,1},{1,3,1}});
        cout << "Test case 4 passed." << endl;
        
        // Test case 5
        vector<vector<int>> result = sol.modifiedGraphEdges(6, {{0,1,-1},{0,2,-1},{1,2,-1},{1,3,-1},{2,4,-1},{3,4,-1},{3,5,-1},{4,5,-1}}, 0, 5, 1000);
        assert(result.size() == 8);
        for (const auto& edge : result) {
            assert(edge[2] >= 1 && edge[2] <= 1000);
        }
        cout << "Test case 5 passed." << endl;
        
        cout << "All tests passed!" << endl;
    }
};

int main() {
    SolutionTest test;
    test.runTests();
    return 0;
}