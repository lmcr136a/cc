import time


start = time.time()
while 1:
    if time.time() - start > 60*60:
        start = time.time()
        
        with open('before_sym.txt', 'w') as f:
            f.write("")