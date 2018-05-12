def gopher_api(req):
    print(req)
    i, j = map(int, input('').split())
    if i == j == 0:
        raise StopIteration
    elif i == j == -1:
        raise ValueError
    else:
        return (i, j)


for case in range(int(input(''))):
    area = int(input(''))
    area = max(area, 9)
    if area % 3 == 1:
        area += 2
    elif area % 3 == 2:
        area += 1

    marked = [[False, False, False], [False, False, False], [False, False, False]]
    current_row = 2
    current_column = 2
    column = 1
    while True:
        if all(marked[0]):
            current_row += 1
            marked.pop(0)
            marked.append([False, False, False])
        try:
            i, j = gopher_api("{} {}".format(current_row, current_column))
            marked[i - current_row + 1][j - 1] = True
        except StopIteration:
            break
        except Exception:
            raise
