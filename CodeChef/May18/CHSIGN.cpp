#include <bits/stdc++.h>
using namespace std;
 
#define little long long
 
int k;
int temp;
little S[100005];
little deep[100005][2];
int isstatus[100005];
little rec( int stat, int aquire ){
	if ( stat >= k ) return 0;
	
	little &res = deep[stat][aquire] ;
	if ( res != -1 ) return res;
	res = 1e18;
	
	res = min( res, rec(stat+1,0) + S[stat] );
	if ( aquire ){
		if ( stat+1 >= k || S[stat+1] > S[stat] ){
			if ( -S[stat] + S[stat-1] - S[stat-2] > 0 ){
				res = min( res, rec(stat+2,1) - S[stat] + ((stat+1>=k)?0:S[stat+1]));
			}
		}
	} else {
		if ( S[stat-1] > S[stat] ){
			if ( stat+1 >= k || S[stat+1] > S[stat] ){
				res = min( res, rec(stat+2,1) - S[stat] + ((stat+1>=k)?0:S[stat+1]));
			}
		}
	}
	return res;
}
void dfs( int stat, int aquire ){
	if ( stat >= k ) return;
	
	if ( rec(stat+1,0) + S[stat] == rec(stat,aquire) ) dfs(stat+1,0);
	else {
		isstatus[stat] = 0;
		dfs(stat+2,1);
	}
}
int main(){	
	ios_base::sync_with_stdio(0);
	cin.tie(0);
	cout.tie(0);	
	cin >> temp;
	while ( temp-- ){
		cin >> k ;
		for ( int i = 0; i < k; i++ ) {
			cin >> S[i];
			deep[i][0] = deep[i][1] = -1;
			isstatus[i] = 1;
		}		
		little res = rec(1,0) + S[0];
		if ( k > 1 && S[0] < S[1] ) res = min( res, rec(2,1) - S[0] + S[1] );
		
		if ( rec(1,0) + S[0] == res ) dfs(1,0);
		else {
			isstatus[0] = 0;
			dfs(2,1);
		}		
		for ( int i = 0; i < k; i++ ){
			if ( isstatus[i] ) cout << S[i];
			else cout << -S[i];
			if ( i < k-1 ) cout << " " ;
			else cout << endl;
		}
	}	
	return 0;
} 
