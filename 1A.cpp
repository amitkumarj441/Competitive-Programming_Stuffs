#include<iostream>
int main()
{
    long long i,j,k,num;
    std::cin>>i>>j>>k;
    num=((i+k-1))/k*((j+k-1)/k);
    std::cout<<num;
    return 0;
}
