import sys

INF = 1 << 32

def main():
    cases = int(sys.stdin.readline())
    for case in range(cases):
        n = int(sys.stdin.readline())
        counts = [0]*n
        used = [0]*n
        for i in range(n):
            nums = [int(a) for a in sys.stdin.readline().split()]
            if nums == [-1]:
                return
            prefs = nums[1:]
            
            seen = INF
            nl = -1
            for pref in prefs:
                counts[pref] += 1
                if counts[pref] < seen and used[pref] == 0:
                    seen = counts[pref]
                    nl = pref

            
            if nl != -1:
                used[nl] = 1
                
            print(nl)
            sys.stdout.flush()
    return
            
            
main()
