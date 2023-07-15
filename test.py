from utils import *

b = get_binance()


balance = b.fetch_balance()
positions = balance['info']['positions']
li = []
for position in positions:
    li.append(position["symbol"].replace("USDT", "/USDT"))
print(len(li), li[0])

with open("syms_pre.txt", 'r') as f:
    sl1 = eval(f.read())
with open("syms.txt", 'r') as f:
    sl2 = eval(f.read())

print(len(sl1), len(sl2))

real_list = []
for s2 in sl2:
    if s2 in sl1 and s2 in li:
        real_list.append(s2)

print(len(real_list))

with open("symlist.txt", 'w') as f:
    f.write(str(real_list))