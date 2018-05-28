#!/bin/python

import os
import sys

n , q = map(int , raw_input().split())

grps = {}
n_map = {}
loop = 0


def countP(a, k):
    n = len(a)
    a.sort()
    res = 0
    for i in xrange(n): 
        j = i+1
        while (j < n and a[j] - a[i] < k):
            j += 1
        res += (n - j)
    return res

for i in xrange(1,n+1):
    grps[loop] = [i]
    n_map[i] = loop
    loop += 1  

for i in xrange(q):
    data = map(int , raw_input().split())
    if(data[0] == 1):
        one , two = data[1] , data[2]
        pos_one = n_map[one]
        pos_two = n_map[two]
        if(pos_one == pos_two):continue
        if(len(grps[pos_one]) < len(grps[pos_two]) ):
            grps[pos_two] += grps[pos_one]
            for i in grps[pos_one]:
                n_map[i] = pos_two
            del grps[pos_one]
        else:
            grps[pos_one] += grps[pos_two]
            for i in grps[pos_two]:
                n_map[i] = pos_one
            del grps[pos_two]
    else:
        c = data[1]
        t_len_list = [len(x) for x in grps.values()]
        print countP(t_len_list,c)
