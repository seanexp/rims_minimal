
# import copy
from functools import partial
from pprint import pprint
from collections import defaultdict
from typing import Union, Tuple
from pathlib import Path 



import datasets
from fire import Fire
from tqdm import tqdm
import jsonlines as jsl
import pandas as pd 

from llm_query_utils import *
from plancode_util_v2 import *






def sln_eval(sln:str='', ans:Union[float, int]=-9999., method:str='')->Tuple[bool, Union[float, int]]:
    # eval logic same as model-selection paper original code
    assert sln
    assert method in ['cot', 'pal', 'p2c']
    assert isinstance(sln, str)
    assert ans!=-9999., 'ans kwarg required'
    
    try:
        if method == 'cot':
            pred = extract_num_turbo(sln)
        elif method == 'pal': #pal, p2c
            pred = safe_execute_turbo(sln)
        else: # p2c
            pred = safe_execute_turbo(postprocess_code(sln)) # postprocess_code will remove backticks from the code. This is applied only for p2c, I keep that bias in order to keep the experiments' consistency
    except:
        pred = None
        
    iscorrect = False
    if pred is None:
        return iscorrect, None # eval, pred
    else: # pred worked!
        try:
            iscorrect = (abs(pred-ans) < 1e-3)
        except Exception as e:
            print(e)
        return iscorrect, pred # eval, pred 



def main(
    config_f:str = '0_1_rims_gather_config_gpt4turbo_fix_p2c.yaml',
):
    # this code only modifies the faulty part of the p2c in the resolved.


    print(f'loaded config from:\n\t{config_f}')
    kwargs = yaml.full_load(open(config_f))
    pprint(kwargs)
    # unpack kwargs
    for k,v in kwargs.items():
        globals()[k] = v
        # now kwargs[k] is accessible as a global variable `k` in this script    
    
    # check are they loaded properly 
    if not Path(dataset_f).is_file():
        gsm8k_train_download_and_parse()
        assert Path(dataset_f).is_file(), 'why is it not downloaded?'

    # sort and sample trainset (no length bias)
    assert fix_p2c_only
    jslf = '/Users/seonils/dev/llm-reasoners/examples/Model-Selection-Reasoning/src/portable/rims_train_out/nov25/resolved_p2c_n3_numtrain1000_gpt4turbo_vT1.0_cT1.0_seed777.jsonl'
    train_samples = list(jsl.open(jslf))


    if dbg:
        train_samples = train_samples[:1]
    
    # for cot, pal, and p2c, find the examples that successes reflection and resolution.
    method = 'p2c'
    method2query_f = {
        'cot': query_cot,
        'pal': query_pal,
        'p2c': query_plancode, 
    }
    f = method2query_f[method]
    kwargs = dict(plan_temperature=verbal_T, code_temperature=code_T, backbone=backbone, n=n_llmquery, seed=seed)
    query_f = partial(f, **kwargs)

    # find the ones that fails at first.
    to_throw_idx = []
    for i, row in tqdm(enumerate(train_samples), desc=method, total=len(train_samples)):
        try:
            ans = row['ans']
            # row = copy.deepcopy(row) 
            out = do_with_tenacity(query_f, row['question']) 
            
            slnlst, planlst, querymsg_d = out
            plan = planlst.pop() # there's only one plan for even n>1
            eval_pred_lst = [sln_eval(sln=sln, ans=ans, method=method) for sln in slnlst]
            eval_lst = [eval_ for eval_, pred in eval_pred_lst if pred is not None] # drop when answer wasn't parseable

            if eval_lst.count(False) <= eval_lst.count(True):
                to_throw_idx.append(i)
                continue # became less valuable 
            old_freq = row['fail_freq']
            new_freq = f"{backbone}: {eval_lst.count(False)}/{n_llmquery}"
            if new_freq != old_freq:
                print(f'{i}: old_(fail)freq changed!')
                print(old_freq)
                print('this changes to...')
                print(new_freq)
                row['fail_freq'] = new_freq
            
            row['fail_preds'] = [eval_pred[-1] for eval_pred, eval in zip(eval_pred_lst, eval_lst) if not eval]
            # only failed solutio gathered
            row['good_solutions'] = [f"{plan}\n{sln}".strip() for sln, e in zip(slnlst, eval_lst) if (e) and (sln is not None)]
            row['fail_solutions'] = [f"{plan}\n{sln}".strip() for sln, e in zip(slnlst, eval_lst) if (not e) and (sln is not None)]
            
            row['fail_method'] = method
            row['llmquery_kwargs'] = kwargs
        except Exception as e:
            print(e)
            continue



    outdir_ = Path(outdir)
    if not outdir_.exists():
        outdir_.mkdir(parents=True, exist_ok=True)
    fname = f"fixed_resolved_p2c_n3_numtrain1000_gpt4turbo_vT1.0_cT1.0_seed777.jsonl"
    if dbg:
        fname = f"dbg_{fname}"
    outpath = outdir_/fname
    with jsl.open(outpath, 'w') as writer:
        writer.write_all(train_samples)
        print(f"{method} failure cases written to:\n\t{str(outpath)}")
        
        
            

            



# util for gsm8k train split download and parsing
def gsm8k_train_download_and_parse(root:str='./'):
    # check if exists
    root = Path(root)
    target_path = root/"gsm8k_train.jsonl"
    if target_path.exists():
        records = list(jsl.open(target_path))
        print(f"found train set @:\n\t{str(target_path)}")
        print(f"\t{records[0]=}")
        print(f"\t{len(records)=}")
    else: 
        # download 
        gsm_train = datasets.load_dataset('gsm8k', 'main')['train']

        # parse
        def parse_raw_target(answer_raw:str)-> str:
            ans_str = answer_raw.split("### ")[-1].strip().replace(",","_")
            try:
                ans = float(ans_str)
            except:
                ans = ans_str
            return ans
        df = pd.DataFrame(gsm_train)
        df['ans'] = df.answer.apply(parse_raw_target)
        df['idx'] = df.index
        records = df.to_dict(orient='records')
        with jsl.open(target_path, 'w') as writer:
            writer.write_all(records)
            print(f'gsm8k train download and write: done.\n\t{str(target_path)}')
    return str(target_path)

if __name__ == "__main__":
    Fire(main)