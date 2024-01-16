
from pathlib import Path
import re 
import subprocess as sb 

pfs = list(Path().glob('*.txt'))
pfs = [p for p in pfs if p.name != 'openai_key.txt']

newfs = [str(f).replace('.txt', '.txt_rm_ans') for f in pfs]

def erase_answer_hinting(prompt:str)->str:
    ptn = r'\(correct answer: [0-9]+[\.]*[0-9]*\)'
    matches = re.findall(ptn, prompt)
    # [print(m) for m in matches]
    # print()
    for txt in matches:
        prompt = prompt.replace(txt, "")
    return prompt

for f, nf in zip(pfs, newfs):
    prompt = open(f).read().strip()
    newp = erase_answer_hinting(prompt)
    with open(nf, 'w') as wf:
        wf.write(newp)
        print(nf)

# mv original ans containing prompts to bak_eval_w_ans_prompts
(Path()/'bak_eval_w_ans_prompts').mkdir(exist_ok=True)
for f in pfs:
    cmd = f'mv {f} bak_eval_w_ans_prompts/'
    print(cmd)
    sb.call(cmd, shell=True)    
    