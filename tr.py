import itertools

C = []
for x in range(2):
    c1= [0,1]
    c2 = [5,1]
    c3= [2,2]
    C = list(itertools.chain(C, [c1], [c2], [c3]))

for y in C:
    if y in C:
        C.remove(y)

print(C)