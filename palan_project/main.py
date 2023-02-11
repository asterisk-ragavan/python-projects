inp = input()
d=0
de = len(inp)-2
temp = []
ver = []
for i in range(de):
    temp = 0
    for j in inp:
        if(j!=de+2):
            ver.append(int(j))
            de= de-1
            d=d+1
    temp.append(ver.reverse())
    if(j==de):
        temp.append(int(j))
    inp=temp
print(temp[len(temp)-1])