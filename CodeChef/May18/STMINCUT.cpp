#include <bits/stdc++.h>
using namespace std;
 
#define little long long
 
struct axel{
	int u,v;
	little w;
};
 
int k;
int temp;
little cost[1005][1005];
little ecost[1005][1005];
little result;
int dep[1005];
vector < int > stre[1005];
vector < axel > f;
 
bool cmp( axel a, axel b ){
	return a.w > b.w;
}
int fin( int x ){
	if ( dep[x] != x ) return dep[x] = fin(dep[x]);
	return x;
}
little merge( int u ,int v, little w ){
	int pu = fin(u), pv = fin(v);
	little res = 0;
 
	for ( int i = 0; i < stre[pv].size(); i++ ){
		for ( int j = 0; j < stre[pu].size(); j++ ){
			int x = stre[pv][i], y = stre[pu][j];
			res += abs(cost[x][y] - w );
			ecost[x][y] = ecost[y][x] = w;
		}
	}	
	for ( int i = 0; i < stre[pv].size(); i++ ){
		stre[pu].push_back(stre[pv][i]);
	}
	stre[pv].clear();
	dep[pv] = pu;
	return res;
}
little mincut(){
	f.clear();
	for ( int i = 1; i <= k; i++ ){
		for ( int j = i+1; j <= k; j++ ){
			axel e;
			e.u = i;
			e.v = j;
			e.w = cost[i][j];
			f.push_back(e);
		}
	}	
	sort(f.begin(),f.end(),cmp);	
	little res = 0;
	for ( int i = 0; i < f.size(); i++ ){
		axel e = f[i];
		int u = e.u, v = e.v;
		little w = e.w;
 
		if ( fin(u) != fin(v) ){
			res += merge(u,v,w)*2;
		} 
	}
	return res;
}
int main(){	
	ios_base::sync_with_stdio(0);
	cin.tie(0);
	cout.tie(0);	
	cin >> temp;
	while ( temp-- ){
		cin >> k;
		for ( int i = 1; i <= k; i++ )
			for ( int j = 1; j <= k; j++ ){
				cin >> cost[i][j];
				ecost[i][j] = -1;
			}			
		result = 0;
		for ( int i = 1; i <= k; i++ )
			for ( int j = 1 ; j <= k; j++ ){
				if ( cost[i][j] < cost[j][i] ){
					result += cost[j][i] - cost[i][j];
					cost[i][j] = cost[j][i];
				} else {
					result += cost[i][j] - cost[j][i];
					cost[j][i] = cost[i][j];
				}
			}			
		for ( int i = 1; i <= k; i++ ){
			dep[i] = i;
			stre[i].clear();
			stre[i].push_back(i);
		}		
		cout << result + mincut() << endl;
	}	
	return 0;
} 
