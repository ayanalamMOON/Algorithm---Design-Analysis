/*
 * Title: Multi-dimensional Tetris Piece Fitter
 * 
 * Description:
 * In this challenge, you need to determine if a given n-dimensional tetris piece can fit into
 * a provided n-dimensional space. Each tetris piece is represented as a set of connected unit cubes
 * in an n-dimensional space, and you need to check if it can be placed (possibly with rotations)
 * into the target space.
 * 
 * The piece can be rotated in any of the 2^n * n! possible orientations (all combinations of reflections
 * and rotations in n-dimensions).
 * 
 * Input:
 * - An n-dimensional array representing the target space (0 = empty, 1 = filled)
 * - An n-dimensional array representing the tetris piece (1 = part of the piece, 0 = empty space)
 * 
 * Output:
 * - true if the piece can fit into the target space, false otherwise
 * 
 * Example (2D):
 * Target space:
 * [
 *   [0, 0, 0],
 *   [0, 0, 0],
 *   [1, 1, 0]
 * ]
 * 
 * Tetris piece:
 * [
 *   [1, 1],
 *   [1, 0]
 * ]
 * 
 * Output: true (it can fit in the upper right corner after a 90-degree rotation)
 * 
 * Constraints:
 * - 1 ≤ n ≤ 4 (dimensions)
 * - Each dimension of the target space and tetris piece is ≤ 10
 * - The tetris piece will always be a single connected component
 */
#include <bits/stdc++.h>
using namespace std;

static const int MAXN = 4;

struct NDPoint {
    int coord[MAXN];
};

class Solution {
public:
    bool canFit(vector<int>& spaceDims,
                const vector<int>& spaceData,
                vector<int>& pieceDims,
                const vector<int>& pieceData,
                int n) {
        vector<NDPoint> pieceCoords;
        {
            vector<int> idx(n, 0);
            while (true) {
                int linear = 0, stride = 1;
                for (int i = n - 1; i >= 0; i--) {
                    linear += idx[i] * stride;
                    stride *= pieceDims[i];
                }
                if (pieceData[linear] == 1) {
                    NDPoint p;
                    for (int i = 0; i < n; i++) p.coord[i] = idx[i];
                    pieceCoords.push_back(p);
                }
                int pos = n - 1;
                while (pos >= 0 && ++idx[pos] == pieceDims[pos]) {
                    idx[pos] = 0; 
                    pos--;
                }
                if (pos < 0) break;
            }
        }
        
        vector<vector<array<int,MAXN>>> allBases;
        {
            vector<int> perm(n);
            iota(perm.begin(), perm.end(), 0);
            function<void(int)> genPerm = [&](int k) {
                if (k == n) {
                    int limit = 1 << n;
                    for (int mask = 0; mask < limit; mask++) {
                        vector<array<int,MAXN>> basis(n, {0});
                        for (int i = 0; i < n; i++) {
                            int s = ((mask & (1 << i)) == 0) ? 1 : -1;
                            basis[i][perm[i]] = s;
                        }
                        allBases.push_back(basis);
                    }
                    return;
                }
                for (int i = k; i < n; i++) {
                    swap(perm[k], perm[i]);
                    genPerm(k+1);
                    swap(perm[k], perm[i]);
                }
            };
            genPerm(0);
        }
        
        auto inBounds = [&](const NDPoint& p) {
            for (int i = 0; i < n; i++) {
                if (p.coord[i] < 0 || p.coord[i] >= spaceDims[i]) return false;
            }
            return true;
        };
        
        auto spaceAt = [&](const NDPoint& p) {
            long long linear = 0, stride = 1;
            for (int i = n - 1; i >= 0; i--) {
                linear += p.coord[i] * stride;
                stride *= spaceDims[i];
            }
            return spaceData[linear];
        };
        
        for (auto& basis : allBases) {
            vector<NDPoint> transformed;
            transformed.reserve(pieceCoords.size());
            for (auto &pc : pieceCoords) {
                NDPoint np = {};
                for (int d = 0; d < n; d++) {
                    for (int x = 0; x < n; x++) {
                        np.coord[d] += pc.coord[x] * basis[d][x];
                    }
                }
                transformed.push_back(np);
            }

            NDPoint mn, mx;
            for (int d = 0; d < n; d++) {
                mn.coord[d] = INT_MAX;
                mx.coord[d] = INT_MIN;
            }
            for (auto &v : transformed) {
                for (int d = 0; d < n; d++) {
                    mn.coord[d] = min(mn.coord[d], v.coord[d]);
                    mx.coord[d] = max(mx.coord[d], v.coord[d]);
                }
            }

            vector<int> pieceSize(n);
            for (int d = 0; d < n; d++) {
                pieceSize[d] = mx.coord[d] - mn.coord[d] + 1;
                if (pieceSize[d] > spaceDims[d]) {
                    goto skipTransform;
                }
            }
            
            {
                vector<int> from(n), to(n);
                for (int d = 0; d < n; d++) {
                    from[d] = 0;
                    to[d] = spaceDims[d] - pieceSize[d];
                }
                
                bool canPlace = false;
                function<void(int)> placeCheck = [&](int dim) {
                    if (dim == n) {
                        for (auto &t : transformed) {
                            NDPoint p = {};
                            for (int dd = 0; dd < n; dd++) {
                                p.coord[dd] = t.coord[dd] - mn.coord[dd] + from[dd];
                            }
                            if (!inBounds(p) || spaceAt(p) == 1) {
                                return;
                            }
                        }
                        canPlace = true;
                        return;
                    }
                    for (int shiftVal = from[dim]; shiftVal <= to[dim]; shiftVal++) {
                        int old = from[dim];
                        from[dim] = shiftVal;
                        placeCheck(dim+1);
                        if (canPlace) return;
                        from[dim] = old;
                    }
                };
                placeCheck(0);
                if (canPlace) return true;
            }
            skipTransform:;
        }
        return false;
    }
};

int main(){
    Solution sol;
    vector<int> spaceDims1 = {3, 3};
    vector<int> spaceData1 = {
        0, 0, 0,
        0, 0, 0,
        1, 1, 0
    };
    vector<int> pieceDims1 = {2, 2};
    vector<int> pieceData1 = {
        1, 1,
        1, 0
    };
    cout << "Test case 1: " << (sol.canFit(spaceDims1, spaceData1, pieceDims1, pieceData1, 2) ? "true" : "false") << endl;
    
    vector<int> spaceDims2 = {3, 3};
    vector<int> spaceData2 = {
        0, 0, 0,
        0, 1, 1,
        0, 1, 1
    };
    vector<int> pieceDims2 = {3, 1};
    vector<int> pieceData2 = {
        1, 1, 1
    };
    cout << "Test case 2: " << (sol.canFit(spaceDims2, spaceData2, pieceDims2, pieceData2, 2) ? "true" : "false") << endl;
    
    vector<int> spaceDims3 = {3, 3};
    vector<int> spaceData3 = {
        0, 0, 1,
        0, 0, 1,
        1, 1, 1
    };
    vector<int> pieceDims3 = {3, 2};
    vector<int> pieceData3 = {
        0, 1, 0,
        1, 1, 1
    };
    cout << "Test case 3: " << (sol.canFit(spaceDims3, spaceData3, pieceDims3, pieceData3, 2) ? "true" : "false") << endl;
    
    vector<int> spaceDims4 = {3, 3, 3};
    vector<int> spaceData4(27, 0);  
    vector<int> pieceDims4 = {2, 2, 2};
    vector<int> pieceData4(8, 1);   
    cout << "Test case 4: " << (sol.canFit(spaceDims4, spaceData4, pieceDims4, pieceData4, 3) ? "true" : "false") << endl;
    
    vector<int> spaceDims5 = {2, 2, 3};
    vector<int> spaceData5 = {
        0, 0,  0, 0,  0, 0,
        0, 0,  0, 0,  0, 0
    };
    vector<int> pieceDims5 = {1, 1, 3};
    vector<int> pieceData5 = {1, 1, 1};
    cout << "Test case 5: " << (sol.canFit(spaceDims5, spaceData5, pieceDims5, pieceData5, 3) ? "true" : "false") << endl;
    
    return 0;
}