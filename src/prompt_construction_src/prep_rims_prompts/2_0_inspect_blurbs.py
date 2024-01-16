import json
from itertools import permutations 
m = 'cot pal p2c'.split()
keys = ['2'.join(p) for p in permutations(m,2)]
 
d = json.load(open('2_reflect_once_blurbs.json'))
for k in keys:
    for i, ex in enumerate(d[k],1):
        print(ex);print(k);print(f"{i}/{len(d[k])}");input()