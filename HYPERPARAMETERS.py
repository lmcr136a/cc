RISING = "Rising"
FALLING = "Falling"
LONG = "Long"
SHORT = "Short"

CALM = 0.07 # 20배일때 1%

with open("symlist.txt", 'r') as f:
    SYMLIST = eval(f.read())

COND_LV1, COND_LV2 = 0.35, 0.2
SATISFYING_LV1, SATISFYING_LV2 = 6, 3  # 이만큼만 먹고 나오기
