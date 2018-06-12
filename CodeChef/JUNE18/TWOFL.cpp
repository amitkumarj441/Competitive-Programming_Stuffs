//Author: Amit Kumar Jaiswal

#include <bits/stdc++.h>

using namespace std;
 
int LEGEND(vector<vector<int>>  M, int row, int col, vector<vector<bool>>& visited)
{
    return (row >= 0) && (row < M.size()) &&     
           (col >= 0) && (col < M[0].size()) &&      
           (M[row][col] && !visited[row][col]); 
}
 
int DEFM(vector<vector<int>>  M, int row, int col, vector<vector<bool>>& visited, int & cnt)
{
    static int rowNbr[] = { -1, 0, 0, 1};
    static int colNbr[] = { 0, -1, 1, 0};
 
    visited[row][col] = true;
 
    for (int k = 0; k < 4; ++k)
        if (LEGEND(M, row + rowNbr[k], col + colNbr[k], visited) ) {
            cnt ++;
            DEFM(M, row + rowNbr[k], col + colNbr[k], visited, cnt); 
        }
    return cnt;
}
 
int countInfinite(vector<vector<int>>  M)
{
    vector<vector<bool>> visited(M[0].size(), vector<bool>(M.size(), false));
    int max_cnt = 0;
    for (int i = 0; i < M.size(); ++i)
        for (int j = 0; j < M[0].size(); ++j)
            if (M[i][j] && !visited[i][j]) 
            {                         
                int cnt = 1;
                DEFM(M, i, j, visited, cnt);
                if (cnt > max_cnt)
                {
                    max_cnt = cnt;
                }
            }
 
    return max_cnt;
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    int x, y;
    cin >> x;
    cin >> y;
    vector<vector<int>> cur(101, vector<int>(101, 0));
    vector<vector<int>> v(x, vector<int>(y));
    int z;
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            cin >> z;
            v[i][j] = z;
        }
    }
    int ans = 0;
    int maxans = 0;
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y-1; j++) {
            if (i <= 100 && j <= 100 &&  cur[v[i][j]][v[i][j+1]] == 1) {
                continue;
            }
            if (i <= 100 && j <= 100) {
                cur[v[i][j]][v[i][j+1]] = 1;
                cur[v[i][j+1]][v[i][j]] = 1;
            }
            vector<vector<int>> d(x, vector<int>(y, 0));
            for (int l = 0; l < x; l++) {
                for (int g = 0; g < y; g++) {
                    if (v[l][g] == v[i][j]) 
                        d[l][g] = v[i][j];
                    if(v[l][g] == v[i][j+1] )
                        d[l][g] = v[i][j+1];
                }
                
            }
            ans = countInfinite(d);
            if (ans > maxans) {
                maxans = ans;
            }
        }
    }
    for (int i = 0; i < x - 1; i++) {
        for (int j = 0; j < y; j++) {
            if (i <= 100 && j <= 100 &&  cur[v[i+1][j]][v[i][j]] == 1) {
                continue;
            }
            if (i <= 100 && j <= 100) {
                cur[v[i+1][j]][v[i][j]] = 1;
                cur[v[i][j]][v[i+1][j]] = 1;
            }
            vector<vector<int>> d(x, vector<int>(y, 0));
            for (int l = 0; l < x; l++) {
                for (int g = 0; g < y; g++) {
                    if (v[l][g] == v[i+1][j]) 
                        d[l][g] = v[i+1][j];
                    if(v[l][g] == v[i][j] )
                        d[l][g] = v[i][j];
                }
            }
            ans = countInfinite(d);
            if (ans > maxans) {
                maxans = ans;
            }
        }
    }
    cout << maxans;
	return 0;
} 
