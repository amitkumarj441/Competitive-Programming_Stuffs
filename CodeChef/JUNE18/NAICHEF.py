#Author: Amit Kumar Jaiswal

am_test = int(input())
 
for i in range(am_test):
    am_dice_sides, dicex, dicey = map(int, input().split())
 
    list_dice_sides = list(map(int, input().split()))
 
    prob_x = list_dice_sides.count(dicex)/am_dice_sides
    prob_y = list_dice_sides.count(dicey)/am_dice_sides
 
    prob_win = prob_x*prob_y    
 
    print("%.10f" %prob_win) 
