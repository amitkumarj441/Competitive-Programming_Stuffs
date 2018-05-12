import sys
INF = 1 << 40

def main():
    cases = int(sys.stdin.readline())
    for case in range(cases):
        n = int(sys.stdin.readline())
        w = [int(a) for a in sys.stdin.readline().split()]
        
        print ("Case #{}: {}".format(case+1,ants(n,w)))
        
        
                
def ants(n,ws):
    out = [INF]*n
    ans = 1
    for i in range(n):
        if i == 0:
            out[i] = ws[i]
        else:
            for j in reversed(range(ans)):
                if out[j] <= 6*ws[i]:
                    out[j+1] = min(out[j+1], out[j]+ws[i])
                    ans = max(j+2, ans)
            out[0] = min(out[0],ws[i])
    return ans
