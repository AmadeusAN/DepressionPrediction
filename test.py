from itertools import *

a = [1, 2, 3]

# for x, y, z in permutations(a, r=3):
#     print(x, y, z)

a_per = [a[x] + a[y] + a[z] for x, y, z in permutations(range(3), r=3)]
print(a_per)
