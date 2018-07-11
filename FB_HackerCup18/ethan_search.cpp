#include<bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
	int temp;
	cin >> temp;
	cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	for (auto m = 1; m <= temp; m++) {
		cout << "Case #" << m << ": ";
		string wd;
		getline(cin, wd);
		int len = wd.length();
		vector<int> head;
		head.reserve(len);
		head.push_back(0);
		for (int p = 1; p < len; p++) {
			if (wd[0] == wd[p]) head.push_back(p);
		}
		int hlen = head.size();
		if (1 == hlen || len == hlen) {
				cout << "Impossible" << endl;
		} else {
			bool possible = false;
			int p = hlen - 1, hjlen;
			while (!possible && 0 < p) {
                                hjlen = len - head[p];
				for (int q = 1; !possible && q < hjlen; q++) possible = wd[head[p]+q] != wd[q];
				p--;
			}
			if (possible) 	cout << wd.substr(0,head[p+1])+ wd << endl;
			else cout << "Impossible" << endl;
		}
	}
	return 0;
}
