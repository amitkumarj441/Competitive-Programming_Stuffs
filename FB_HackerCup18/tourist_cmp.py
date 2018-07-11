#Author: Amit Kumar Jaiswal 

fbcampus_cnt = int(input())
result_case = ""
for i in range(fbcampus_cnt):
    n_k_v = list(map(int, input().split()))
    no_of_atc = n_k_v[0]
    view_atc_per_tour = n_k_v[1]
    tour_cnt = n_k_v[2]
    atc = list()
    for j in range(no_of_atc):
        atc.append(input())
    ps_idx = 0
    ps_idx += view_atc_per_tour * (tour_cnt - 1)
    result_list = [""] * no_of_atc
    result = ""
    for k in range(view_atc_per_tour):
        result_list[ps_idx % no_of_atc] = atc[ps_idx % no_of_atc]
        ps_idx += 1
    for attr in result_list:
        if attr != "":
            result += " "
            result += attr
    result_case += "Case #"+str(i+1)+":"+result+"\n"


print(result_case)
