#include<bits/stdc++.h>
using namespace std;
#define little long long int 
 
int main(){
	ios_base::sync_with_stdio(false);
	 cin.tie(NULL);
	
	int x;cin>>x;
	while(x--){
		int n,m;cin>>n;
		little res=0;
		for(int i=0;i<n;++i) cin>>m,res=res^(2*m);
		cout<<res<<"\n";
	
	}	
	return 0;
} 
