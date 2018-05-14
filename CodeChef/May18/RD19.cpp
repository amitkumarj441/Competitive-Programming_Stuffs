#include <iostream>
#include <algorithm>
using namespace std;
#define space 1001
 
int main(){
    int x,O[space],i;
    cin>>x;
    while(x--){
        int p;
        cin>>p;
        for(i=0; i<p; i++){
            cin>>O[i];
        }
        sort(O,O+p);
        if(O[0]==1)
            cout<<"0"<<endl;
        else{
            int flag=0;
            for(i=1; i<p; i++){
                if(O[i]%O[0]!=0){
                    flag=1;
                    break;
                }
            }
            if(flag==0)
                cout<<"-1"<<endl;
            else
                cout<<"0"<<endl;
        }
    }
    return 0;
} 
