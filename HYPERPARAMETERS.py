RISING = "Rising"
FALLING = "Falling"
LONG = "Long"
SHORT = "Short"
DECREASING_CONVEX = 'Decreasing and Convex'
INCREASING_CONCAVE = 'Increasing and Concave'


CALM = 0.07 # 20배일때 1%

with open("symlist.txt", 'r') as f:
    SYMLIST = eval(f.read())

COND_LV1, COND_LV2 = 0.25, 0.15
SATISFYING_LV1, SATISFYING_LV2 = 15, 15  # 이만큼만 먹고 나오기
