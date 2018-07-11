//Author: Amit Kumar Jaiswal

#include<bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
	int x, temp;
	cin >> temp;
	for (auto i = 1; i <= temp; i++) {
		cout << "Case #" << i << ": ";
		cin >> x;
		int ep, np = 0;
		for (auto j = 0; j < x; j++) {
			cin >> ep;
			if (0 == np) np = 1; 
			else np = 0;
		}
		cin >> ep;
		if (np != 0) {
			cout << 1 << endl;
			cout << 0 << endl;
		}
		else cout << 0 << endl;
	}
	return 0;
}
