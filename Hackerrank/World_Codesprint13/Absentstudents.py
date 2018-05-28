#!/bin/python

import os
import sys

def AbsentStudents(x, a):
    l = []
    for i in range(1, x + 1):
        found = False
        for j in a:
            if i == j:
                found = True

        if found is False:
            l.append(i)
            
    return l

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    x = int(raw_input())
    a = map(int, raw_input().rstrip().split())
    result = AbsentStudents(x, a)
    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')
    fptr.close()
