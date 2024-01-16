from itertools import product, permutations
from collections import Counter 
import json
import yaml
from llm_query_utils import PromptStr
import random
random.seed(777)


TEMPL='3_prompt_templ.yaml'
BLURBS='2_reflect_once_blurbs.json'
N_PROMPTS=9 # multuple of 3
assert N_PROMPTS%3==0, "N_PROMPTS must be a multiple of 3"

# 1. from TEMPL, make a template for the actual RIMS prompt
#   need to take system prompt only, not the others
prompt_src = yaml.full_load(open(TEMPL))
sys = prompt_src['system']
sep = prompt_src['sep']
inst = prompt_src['inst']
prompt = f"{sys}{sep}[BLURB1]{sep}[BLURB2]{sep}[BLURB3]{sep}{inst}"
prompt_tmp = PromptStr(prompt)


# 2. prep directions to make blurbs
#   each method in a prompt need to be once wrong and once correct  
methods = 'cot pal p2c'.split()
correct_candids = methods
wrong_candids = methods


def flatten(lst_pairs)->list:
    flat = []
    for pair in lst_pairs:
        flat.extend(list(pair))
    return flat

def tuplst2strlst(lst_pairs)->list:
    return [f"{tup[0]}2{tup[1]}" for tup in lst_pairs] 

def possible_sequences(correct_candids:list, wrong_candids:list)->list:
    # make all possible pairs
    all_directions = list(product(correct_candids, wrong_candids))
    # remove same2same
    directions = [d for d in all_directions if d[0]!=d[1]]
    # make possible 3-pairs-sequences
    sequences = list(permutations(directions, 3))
    # remove sequences that depicts one method better than the others (i.e. cot is 3 times correct while pal corrects only once)
    sequences = [s for s in sequences if set(Counter(flatten(s)).values())=={2} ]
    # (method1, method2) --> {method1}2{method2}
    directions_fin = [tuplst2strlst(s) for s in sequences]
    return directions_fin

possible_comps = possible_sequences(correct_candids, wrong_candids) # 48 distinct prompt compositions possible... we just pick 9 of those...
# pick 9 of above list with even intervals within same start
starts2possibles = {
    method: [s for s in possible_comps if s[0].startswith(method)] for method in methods
}
picked_directions = {k: v[::len(v)//3][:N_PROMPTS//3] for k,v in starts2possibles.items()} # methods are always 3 so N_PROMPTS//3 * 3 == N_PROMPTS



# 3. make 9 prompts (pick blurb randomly)
d = json.load(open(BLURBS))
for _, sequences in picked_directions.items():
    for seq in sequences:
        pickd = dict()
        for i, dir in enumerate(seq,1):
            blurbs = d[dir]
            pick = random.sample(blurbs, 1)
            pickd[f"BLURB{i}"] = pick.pop().strip()
            actual_prompt = str(prompt_tmp.sub_map(**pickd)) 
        prompt_f = f"3_reflectonce_{'.'.join(seq)}.txt"
        with open(prompt_f, 'w') as f:
            f.write(actual_prompt)
            print(prompt_f)


        



