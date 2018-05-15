#include <bits/stdc++.h>
using namespace std;
 
int a,b;
int temp;
#define LL long long
int s[100005];
vector < int > vc;
map < int, int > ss;
 
int infoq( int X ){
	
	int ans = 0;
	int x = lower_bound(vc.begin(),vc.end(),X) - vc.begin() + 1;
	int big = a - x, small = x - 1;
	int y = ss[X];
	int lo = 1, hi = a, mid;
	int B = 0, S = 0;
	
	while ( lo <= hi ){
		mid = ( lo + hi ) / 2;
		
		if ( s[mid] == X ) break;
		if ( mid < y ){
			if ( s[mid] > X ) S++;
			lo = mid + 1;
			small--;
		} else {
			if ( s[mid] < X ) B++;
			hi = mid - 1;
			big--;
		}
		
		if ( small < 0 || big < 0 ) return -1;
	}
	ans = min( B,S);
	B -= ans;
	S -= ans;
	ans += B + S;
	return ans;
}
int main(){
	
	ios_base::sync_with_stdio(0);
	cin.tie(0);
	cout.tie(0);	
	cin >> temp;
	while ( temp-- ){
		cin >> a >> b;
		vc.clear();
		ss.clear();
		for ( int i = 1; i <= a; i++ ) {
			cin >> s[i];
			vc.push_back(s[i]);
			ss[s[i]] = i;
		}	
		sort(vc.begin(),vc.end());	
		while ( b-- ){
			int x;
			cin >> x;
			cout << infoq(x) << endl;
		}
	}
		return 0;
} 
