'''
using the reflectonce prompt, prep the followings


blurbs per each method:
- reflection*0
- reflection*1
- reflection*2

prompts:
- reflection0 + 1 + 2 (+ordering)
    - 012
    - 021
    - 210
    - ...
- 

if possible, I want every method at least once to be presented as a correct approach (no bias for any method)
but not sure this is really a thing...?

- balanced vs biased
'''

from pathlib import Path
import random
random.seed(777)
import json
from typing import List
from collections import Counter

import jsonlines as jsl
from tqdm import tqdm 

from llm_query_utils import * # query_rims_prompt, PromptStr(class), etc.


CONTINUE_WRITING_INVOKE_PROMPT = "Continue reaching to the correct answer, carefully following the format presented above."
N_EXTEND_TRIALS = 5 #
BACKBONE = 'chatgpt' # gpt4turbo
TEMP = 0.3
BLURB_F = '5_extend_reflection_blurbs.json'
BLURB_PROMPT_F= '5_extend_reflection_blurbs_prompts.json'



# reflect once prompts
reflect_once = list(Path().glob("3_reflectonce_*.txt_rm_ans"))[1:]    

# each prompt will be used for extending blurb it contains 
# train_samples = list(jsl.open('gsm8k_train.jsonl')) # super inefficient as most of the problems will be attempted once and solved
to_gather_files = list(Path('rims_train_out/nov25').glob('resolved*.jsonl')) # instead, perform amongst the reflect-needed ones (used in 1_*.py)
records = [list(jsl.open(f)) for f in to_gather_files]

train_samples = []
for rec in records:
    train_samples.extend(rec)


# helpers  
def did_reflect(response:str)->bool:
    return '`Attempt 2`:' in response

@exception_handler
def is_correct(pred:float, gt:float)->bool:
    return abs(pred-gt)<1e-3

@exception_handler
def query_rims_inference_w_skip(*args, **kwargs):
    return query_rims_inference(*args, **kwargs)

def count_methods_in_blurbs(blurbstr:str)->dict:
    methods = '(cot) (pal) (p2c)'.split()
    countd = dict()
    for m in methods:
        mc = blurbstr.split().count(m)
        countd[m] = mc 
    return countd

def make_prompt(templ_file:str = "3_prompt_templ.yaml",
                blurbs:List[str]=None):
    assert blurbs, f"check {blurbs=}"
    prompt_src = yaml.full_load(open(templ_file))
    sys = prompt_src['system']
    sep = prompt_src['sep']
    inst = prompt_src['inst']
    contents = [sys] + blurbs + [inst]    
    prompt = sep.join(contents)
    # prompt_tmp = PromptStr(prompt)
    return prompt

def extract_blurbs_from_prompt(templ_file:str = "3_prompt_templ.yaml",
                               promptstr:str='')->str:
    assert promptstr, f"check {promptstr=}"
    prompt_src = yaml.full_load(open(templ_file))
    sys = prompt_src['system']
    sep = prompt_src['sep']
    inst = prompt_src['inst']
    blurbs_str = promptstr.replace(sys,"").replace(inst,"").strip()
    blurbs = blurbs_str.split(sep)
    return blurbs

# gather extended blurbs from training set (using the reflectonce prompt)
collected_blurbs = dict()
for pf in tqdm(reflect_once):
    random.shuffle(train_samples)
    
    collected = {
        'cot': [],
        'pal': [],
        'p2c': [],
        'extend': []
    }
    for row in (pbar:=tqdm(train_samples)):
        tqdm_desc = " ".join([f"{k}:{len(v)}/1" for k,v in collected.items()]) + f"    ({pf.stem})"
        # if finish gathering all, break
        pbar.set_description(tqdm_desc)
        if all( [len(v)>1 for v in collected.values()] ):
            break

        # left things to gather
        question = row['question']
        gt = float(row['ans'])
        # check if the question already in the prompt
        if question in pf.open().read().strip():
            print(question)
            print('already in the prompt') # do not allow duplicate questions
            continue
        # query rims inference

        llmout = query_rims_inference_w_skip(question, pf, backbone=BACKBONE, temperature=TEMP)
        if llmout is None:
            continue
        else:
            eval_friendly_d, __, raw_query_out, _ = llmout
        if not raw_query_out.strip().split("\n")[-1].startswith('`Answer '):
            continue # rims inference failed to stop at Evaluation: Correct 
        pred = eval_friendly_d['good_ans']
        
        # chance to extend the blurb
        if not is_correct(pred, gt) and did_reflect(raw_query_out):
            if collected['extend']:
                continue # already done
            # extending blurb
            blurb_1_refl = f"`Question`: {question}\n{raw_query_out.strip()}"
            gpt_messages = [
                {'role': 'assistant', 'content': raw_query_out.strip()},
                {'role': 'user', 'content': CONTINUE_WRITING_INVOKE_PROMPT}
            ]

            llmout = query_rims_inference_w_skip(question, pf, backbone=BACKBONE, temperature=TEMP, n=N_EXTEND_TRIALS, continue_writing_gpt_messages=gpt_messages)
            if llmout is None:
                continue
            else:
                eval_friendly_ds, parse_dds, raw_query_outs, _ = llmout
            
            for i, d in enumerate(eval_friendly_ds):
                newpred = d['good_ans']
                # if newpred == pred: # this will filter "only generating `Evaluation`: Correct" case
                #     print('re_reflection did not change the prediction')
                #     continue
                # else:
                if is_correct(newpred, gt):
                    toappend = raw_query_outs[i]
                    extended = f"{blurb_1_refl.strip()}\n{toappend.strip()}\n`Evaluation`: Correct"
                    collected['extend'].append(extended)
                    break # only one extended blurb per prompt. added. finished for this prompt.
            
        # chance to gather one-shot correct question     
        elif is_correct(pred, gt) and not did_reflect(raw_query_out):
            # gather one-shot blurb
            method = eval_friendly_d['good_method']
            if len(collected[method])>=4: # in real use, we only use first, but just in case lets get 3 more
                print(f'already gathered all one-shot blurb {method}')
                continue
            blurb_1 = f"`Question`: {question}\n{raw_query_out.strip()}\n`Evaluation`: Correct"
            collected[method].append(blurb_1)
        else: # other cases, pass
            # did not reflect, wrong 
            # did reflect, correct
            print(raw_query_out.strip())
            print(f"pred: {pred}, gt: {gt}")
            continue 
    collected_blurbs[pf.name] = collected        
# save the blurb collected with promptfile watermark 
with open(BLURB_F, 'w') as jf:
    json.dump(collected_blurbs, jf, indent=4, ensure_ascii=False)
    print(f'{BLURB_F}')



# augment the prompt with collected blurbs 
for stem, mdict in collected_blurbs.items():
    if mdict['extend']:
        # 1 extended only 
        ext_only_prompt = make_prompt(blurbs = mdict['extend'])

    if any([mdict[m] for m in "cot pal p2c".split()]):
        # 2 oneshot + extended 
        method_counter = count_methods_in_blurbs(mdict['extend'])
        least_common_method = method_counter.most_common()[-1][0].replace("(", "").replace(")", "")
        if mdict[least_common_method]:
            oneshotblurbs = mdict[least_common_method][:1]
        else:
            for m in "cot pal p2c".split():
                if mdict[m]:
                    oneshotblurbs = mdict[m][:1]
                    break
        oneshot_ext_prompt = make_prompt(blurbs = oneshotblurbs + mdict['extend'])
        # 3 oneshot + extended + original
        original_prompt=open(f"{stem}.txt_rm_ans").read().strip()
        original_blurbs = extract_blurbs_from_prompt(promptstr = original_prompt)
        blurbs = oneshotblurbs + mdict['extend'] + original_blurbs
        
        everything_prompt = make_prompt(blurbs = blurbs)
        # add to blurbdict
        collected_blurbs[stem + "_1prompt"] = ext_only_prompt
        collected_blurbs[stem + "_13prompt"] = oneshot_ext_prompt
        collected_blurbs[stem + "_132prompt"] = everything_prompt

# save the indiv prompts into a txt file
for k, v in collected_blurbs.items():
    if k.endswith('_prompt'):
        fname = "5_ext_"+k+ ".txt"
        with open(fname, 'w') as wf:
            wf.write(v.strip())
            print(fname)

with open(BLURB_PROMPT_F, 'w') as jf:
    json.dump(collected_blurbs, jf, indent=4, ensure_ascii=False)
    print(f'{BLURB_PROMPT_F}')