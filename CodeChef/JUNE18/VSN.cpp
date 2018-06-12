//Author: Amit Kumar Jaiswal

#include <bits/stdc++.h>
using namespace std;
 
typedef vector <int> vi;
typedef vector <long long> vl;
#define little long long
#define pb push_back  
#define tr(container, iterator) for (typeof(container.begin()) iterator = container.begin(); iterator != container.end(); ++iterator) 
#define all(c) c.begin(), c.end()
 
little modulo = 1000000007;
 
void printarr(int arr[], int n)
{
    for (int i = 0; i < n; ++i)
    {
        cout << arr[i] << " ";
    }
    cout << endl;
}
 
double p[3],q[3],d[3],c[3];
double r;
template <class X>
 
double solve(X t)
{
    double a[3] = {q[0] + t*d[0], q[1] + t*d[1], q[2] + t*d[2]};
    double b[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};    
    double num = 0.0;
    for (int i = 0; i < 3; ++i)    
        num += ((a[i] - c[i]) * b[i]);
    double den = 0.0;
    for (int i = 0; i < 3; ++i)    
        den += ((a[i] - p[i]) * b[i]);
    double k = num / den;
    double x = k*p[0] - c[0] + (1-k)*a[0];
    double y = k*p[1] - c[1] + (1-k)*a[1];
    double z = k*p[2] - c[2] + (1-k)*a[2];
    double dis = pow(x , 2) + pow(y, 2) + pow(z, 2);
    return dis;
}
 
int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    #ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
    freopen("error.txt", "w", stderr);
    #endif
    time_t t1, t2;
    t1 = clock();
    int t;
    cin >> t;
    while (t--)
    {
    	cin >> p[0]; cin >> p[1]; cin >> p[2];
    	cin >> q[0]; cin >> q[1]; cin >> q[2];
    	cin >> d[0]; cin >> d[1]; cin >> d[2];
    	cin >> c[0]; cin >> c[1]; cin >> c[2];
    	cin >> r;
        r = r * r;
    	int high = INT_MAX;
    	int low = 0;
        double ans;
        while (low <= high)
        {
            double mid = (low + high) / 2;
            double x = solve(mid);
            if (x < r)
                low = mid + 1;
            else
            {
                ans = mid;
                high = mid - 1;
            }
        }
        ans = max(1.0, ans);
        double y;
        for (int i = 0; i < 10; ++i)
        {
            y = ans - pow(0.1,i);
            for (int j = 0; j < 10; ++j)
            {
                if (solve(y + j*pow(0.1,i+1)) >= r)
                {
                    ans = y + j*pow(0.1,i+1);
                    break;
                }
            }
        }
        cout << fixed << setprecision(9) << ans << endl;
    }
    t2 = clock();
    cerr << "time taken: " << (double) (t2 - t1) / (CLOCKS_PER_SEC) * 1000;
    return 0;
} 
