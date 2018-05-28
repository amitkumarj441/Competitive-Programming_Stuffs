#include <bits/stdc++.h>

using namespace std;

vector<string> split_str(string);
const int max_n = 1e5 + 1e2;

std::vector<int> lim;
bool within_limits(int size[]){
    if(size[0] <= lim[1]
                and size[1] <= lim[2] 
                and size[2] <= lim[3] 
                and size[3] <= lim[4]){
        return true;
    }
    return false;
}

struct demsu {
    demsu* parent = this;
    int rank = 0;
    int size[4] = {1, 0, 0, 0};

    demsu* head(){
        if(parent != parent->parent){
            parent = parent->head();
        }
        return parent;
    }
    void merge(demsu* sec){
        demsu* fir = head();
        sec = sec->head();
        int size_sum[] = {fir->size[0] + sec->size[0],
                         fir->size[1] + sec->size[1],
                         fir->size[2] + sec->size[2],
                         fir->size[3] + sec->size[3]};

        if(within_limits(size_sum) and fir != sec){
            std::copy(size_sum, size_sum + 4, fir->size);
            std::copy(size_sum, size_sum + 4, sec->size);
            if(fir->rank < sec->rank){
                std::swap(fir, sec);
            }
            if(fir->rank == sec->rank){
                fir->rank++;
            }
            sec->parent = fir;
        }
    }
};

int grade[max_n];
std::map<std::string, int> mapping;
std::map<int, std::string> mappingrev;
demsu demsu_array[max_n];
int curr_index = 0;
int get_ind(std::string s){
    auto ite = mapping.find(s);
    if(ite == mapping.end()){
        mapping[s] = curr_index;
    	mappingrev[curr_index] = s;
        curr_index++;
        return curr_index - 1;
    } else {
    	return ite->second;
    }
}
std::string get_string(int s){
    return mappingrev[s];
}
std::string milg(int n, int m, int a, int b, int f, int s, int t) {
    memset(grade, 0, sizeof(grade));
    mapping.clear();
    mappingrev.clear();
    curr_index = 0;
    lim = {a, b, f, s, t};
    for(int i = 0; i < n; i++){
        std::string curr_student;
        int curr_grade;
        std::cin >> curr_student >> curr_grade;
        demsu_array[get_ind(curr_student)].size[curr_grade]++;
    }
    int res = 0;
    for(int i = 0; i < m; i++){
        std::string student_a, student_b;
        std::cin >> student_a >> student_b;
        int a_id = get_ind(student_a), b_id = get_ind(student_b);
        demsu_array[a_id].merge(&demsu_array[b_id]);
        res = std::max(res, demsu_array[a_id].head()->size[0]);
    }
    if(res < a){
        return "no groups";
    }
    std::vector<std::string> res_strings;
    for(int i = 0; i < n; i++){
        if(demsu_array[i].head()->size[0] == res){
            res_strings.push_back(get_string(i));
        }
    }
    std::sort(res_strings.begin(), res_strings.end());
    std::ostringstream ostr;
    for(int i = 0; i < res_strings.size(); i++){
        ostr << res_strings[i] << '\n';
    }
    return ostr.str();
}

int main()
{
    ios::sync_with_stdio(false);
    string nmabfst_temp;
    getline(cin, nmabfst_temp);
    vector<string> nmabfst = split_str(nmabfst_temp);
    int n = stoi(nmabfst[0]);
    int m = stoi(nmabfst[1]);
    int a = stoi(nmabfst[2]);
    int b = stoi(nmabfst[3]);
    int f = stoi(nmabfst[4]);
    int s = stoi(nmabfst[5]);
    int t = stoi(nmabfst[6]);
    std::cout << milg(n, m, a, b, f, s, t);
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
