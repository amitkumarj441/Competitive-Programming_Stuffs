#Author: Amit Kumar Jaiswal

def binaryShuffle(a,b):
    if a == b :
        return 0
    if b <= 1:
        return -1
    
    binA = bin(a)[2:]
    binB = bin(b)[2:]
    tz = 0
    index = len(binB)-1
    while True:
        if binB[index] == "0":
            tz +=1
            index -=1
        else:
            break
 
    c1A = binA.count("1")
    c1B = binB.count("1")
    if c1B - 1 >= c1A:
        if binA[-1] == "1":
            return c1B-c1A
        else:
            return c1B-c1A + tz
    else:
        if binB[-1] == "1":
            return 2
        else:
            ec1B = c1B - 1 + tz
            if ec1B == c1A:
                return 1
            elif ec1B > c1A:
                return c1B + tz - c1A
            else:
                return 2 
