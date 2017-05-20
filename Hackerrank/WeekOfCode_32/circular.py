#!/bin/python3

import sys

def circularWalk(n, s, t, steps, visited, count):
    if s == t:
        return count
    
    if s in visited:
        return float('inf')
    else:
        visited[s] = True

    if steps[s] == 0:
        return float('inf')
    else:
        check = []
        for i in range(0-steps[s], steps[s]+1, 1):
            new_s = (s+i)%n
            check.append(circularWalk(n, new_s, t, steps, visited, count+1))
        return min(check)
        

n, s, t = input().strip().split(' ')
n, s, t = [int(n), int(s), int(t)]
r_0, g, seed, p = input().strip().split(' ')
r_0, g, seed, p = [int(r_0), int(g), int(seed), int(p)]
visited = {}
steps = {0:r_0}
for i in range(1, n):
    steps[i] = (steps[i-1]*g+seed)%p
    
result = circularWalk(n, s, t, steps, visited, 0)
if result == float('inf'):
    print("-1")
else:
    print(result)
