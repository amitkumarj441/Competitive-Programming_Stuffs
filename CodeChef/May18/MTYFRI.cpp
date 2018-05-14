#include <algorithm>
#include <iostream>
using namespace std;
#define enlg 10001
 
int main(){
    int x,arr[enlg],arr1[enlg],arr2[enlg],i,j;
    cin>>x;
    while(x--){
        int n,k;
        cin>>n>>k;
        for(i=0; i<n; i++)
            cin>>arr[i];
        if(n==1)
            cout<<"NO"<<endl;   
        else{
            int s1=0,s2=0;           
            int S1=0,S2=0;
            for(i=0; i<n; i++){
                if(i%2==0){
                    arr1[s1]=arr[i];
                    s1++;
                    S1=S1+arr[i];
                }
                else{
                    arr2[s2]=arr[i];
                    s2++;
                    S2=S2+arr[i];
                }
            }
            sort(arr1,arr1+s1);
            sort(arr2,arr2+s2);
            if(k==0){
                if(S1>=S2)
                    cout<<"NO"<<endl;
                else
                    cout<<"YES"<<endl;
            }    
            else{
                int flag1= s1-1,flag2=0,k1=k;
                while(flag1>=0 and flag2<s2 and k1!=0 and arr1[flag1]>arr2[flag2]){
                    int temp = arr1[flag1];
                    arr1[flag1]=arr2[flag2];
                    arr2[flag2]=temp;
                    flag1--;
                    flag2++;
                    k1--;                    
                }
                int S1=0,S2=0;
                for(i=0; i<s1; i++)
                    S1=S1+arr1[i];
                for(i=0; i<s2; i++)
                    S2=S2+arr2[i];
                if(S1>=S2)
                    cout<<"NO"<<endl;
                else
                    cout<<"YES"<<endl;
            }           
        }  
    }
    return 0;
} 
