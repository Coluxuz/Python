import math


def progress_bar(progress, total):
    percent = 100 * (progress/float(total))
    bar = '/'*int(percent)+'-'*(100-int(percent))
    print(f"\r|{bar}|{percent:.2f}%",end="\r")
    
num = [x*5 for x in range(2000,3000)]
result = []

for i,x in enumerate(num):
    result.append(math.factorial(x))
    progress_bar(i+1,len(num))