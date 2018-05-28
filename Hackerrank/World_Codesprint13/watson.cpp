#include <bits/stdc++.h>

typedef long long little;
using namespace std;

vector<string> split_str(string);

little inverse(little a, little m){
	if(a == 0){
		return -1;
	}
    little exp = m - 2;
    little prod = 1;
    while(exp){
        if(exp & 1){
            prod = (prod * a) % m;
        }
        a = (a * a) % m;
        exp /= 2;
    }
    return prod;
}
long hmgs(vector<int> A, int m, int given_k) {
    little n = A.size();
    little mod = m;
    little k = given_k;
    vector<little> v(n);
    for(little i = 0; i < n; i++){
        v[i] = A[i] % mod;
    }
    little res = 0;
    if(k == 0){
        little seen = -1;
        for(little i = 0; i < n; i++){
            if(v[i] == 0){
                seen = i;
            }
            res += seen + 1;
        }
        return res;
    }
    std::queue<little> ql, qr;
    ql.push(0), qr.push(n - 1);
    while(!ql.empty()){
        little l = ql.front(), r = qr.front();
        ql.pop(), qr.pop();
        if(l == r){
            if(v[l] == k){
                res++;
            }
            continue;
        }
        little mid = (l + r) / 2;
        std::map<little, little> m;
        little producted = 1;
        for(little i = mid; i >= l; i--){
            producted = (producted * v[i]) % mod;
            m[producted]++;
        }
        producted = 1;
        for(little i = mid + 1; i <= r; i++){
            producted = (producted * v[i]) % mod;
            res += m[(inverse(producted, mod) * k) % mod];
        }
        ql.push(l), qr.push(mid);
        ql.push(mid + 1), qr.push(r);
    }
    return res;
}

int main()
{
    ios::sync_with_stdio(false);
    ofstream fout(getenv("OUTPUT_PATH"));
    int t;
    cin >> t;
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    for (int t_itr = 0; t_itr < t; t_itr++) {
        string nmk_t;
        getline(cin, nmk_t);
        vector<string> nmk = split_str(nmk_t);
        int n = stoi(nmk[0]);
        int m = stoi(nmk[1]);
        int k = stoi(nmk[2]);
        string A_temp_temp;
        getline(cin, A_temp_temp);
        vector<string> A_temp = split_str(A_temp_temp);
        vector<int> A(n);
        for (int i = 0; i < n; i++) {
            int A_item = stoi(A_temp[i]);
            A[i] = A_item;
        }
        long result = hmgs(A, m, k);
        fout << result << "\n";
    }
    fout.close();
    return 0;
}

vector<string> split_str(string input_string) {
    string::iterator new_end = unique(input_string.begin(), input_string.end(), [] (const char &x, const char &y) {
        return x == y and x == ' ';
    });
    input_string.erase(new_end, input_string.end());
    while (input_string[input_string.length() - 1] == ' ') {
        input_string.pop_back();
    }
    vector<string> split;
    char delimiter = ' ';
    size_t i = 0;
    size_t pos = input_string.find(delimiter);

    while (pos != string::npos) {
        split.push_back(input_string.substr(i, pos - i));
        i = pos + 1;
        pos = input_string.find(delimiter, i);
    }
    split.push_back(input_string.substr(i, min(pos, input_string.length()) - i + 1));
    return split;
}
