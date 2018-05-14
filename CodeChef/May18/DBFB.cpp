#include<bits/stdc++.h>
using namespace std;
typedef long long int little;
#define sm 100005
#define mod 1000000007
 
little A[sm],B[sm];
little add(little x,little y){
	little z=x+y;
	if(z>=mod) z-=mod;
	return z;
}
little mul(little x,little y){
	little z=x*y;
	if(z>=mod) z%=mod;
	return z;
}
inline void bef(){
	A[1]=1;B[1]=0;
	A[2]=0;B[2]=1;
	for(int i=3;i<sm;++i){
		A[i]=add(A[i-1],A[i-2]);
		B[i]=add(B[i-1],B[i-2]);
	}
}
int main(){
	 ios_base::sync_with_stdio(false);
	 cin.tie(NULL);
	bef();
	int x;cin>>x;
	while(x--){
		int n,m;
		cin>>m>>n;
		little d;
		little res=0;
		for(int i=0;i<m;++i){
			cin>>d;
			res=add(res,mul(m,mul(A[n],d)));
		}
		for(int i=0;i<m;++i){
			cin>>d;
			res=add(res,mul(m,mul(B[n],d)));
		}
		cout<<res%mod<<"\n";	
	}
	return 0;
}  
