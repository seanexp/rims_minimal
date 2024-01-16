from fire import Fire

import copy
from functools import partial
from pprint import pprint
from typing import Union, Tuple

from llm_query_utils import *







def sln_eval(sln:str='', ans:Union[float, int]=-9999., method:str='')->Tuple[bool, Union[float, int]]:
    # eval logic same as model-selection paper original code
    assert sln
    assert method in ['cot', 'pal', 'p2c']
    assert isinstance(sln, str)
    assert ans!=-9999., 'ans kwarg required'
    
    try:
        if method == 'cot':
            pred = extract_num_turbo(sln)
        else: #pal, p2c
            pred = safe_execute_turbo(sln)
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
    config_f:str = '1_rims_resolve_config.yaml',
):
    print(f'loaded config from:\n\t{config_f}')
    kwargs = yaml.full_load(open(config_f))
    pprint(kwargs)
    # unpack kwargs
    for k,v in kwargs.items():
        globals()[k] = v
        # now kwargs[k] is accessible as a global variable `k` in this script    
    

    # methods of interest
    METHODS = 'p2c pal cot'.split()
    if not do_p2c:
        METHODS.remove('p2c')
    
    # query_functions
    method2query_f = {
        'cot': query_cot,
        'pal': query_pal,
        'p2c': query_plancode, 
    }

    for f_method in METHODS:
        jslf = to_reflect_jsl.replace('METHOD', f_method)
        outjslf = jslf.replace('failed', 'resolved')

        # # quick fixing for p2c buggy files
        # if only_modif_p2c:
        #     jslf = outjslf # buggy files of p2c solutions w/o plan (only codes)
        #     outjslf = jslf.replace(".jsonl", "_fixed_p2c.jsonl") 

        records = list(jsl.open(jslf))

        # re-solve the failed questions with different method
        results = list()

        for row in tqdm(records, desc=f're-solving for failed {f_method}'):
            row = copy.deepcopy(row)
            for method in METHODS:
                # try:
                
                # only attempt p2c for resolution (dec27 modifying bugs in p2c files)
                # if only_modif_p2c:
                #     if method != 'p2c': # only p2c attempt will pass
                #         continue
                
                # reattempting with same method as original?
                if reattempt_same_method:
                    pass
                else:
                    if method == f_method:
                        continue

                # define query function 
                f = method2query_f[method]
                if method == 'cot':
                    kwargs = dict(cot_temperature=verbal_T, backbone=backbone, n=n_llmquery, seed=seed)
                    query_f = partial(f, **kwargs)
                elif method == 'pal':
                    kwargs = dict(pal_temperature=verbal_T, backbone=backbone, n=n_llmquery, seed=seed)
                    query_f = partial(f, **kwargs)
                elif method == 'p2c':
                    kwargs = dict(plan_temperature=verbal_T, code_temperature=code_T, backbone=backbone, n=n_llmquery, seed=seed)
                    query_f = partial(f, **kwargs)
                else:
                    raise ValueError(f'unknown method {method}')  
                out = do_with_tenacity(query_f, row['question']) if not dbg else query_f(row['question'])
                if method == 'p2c':
                    slnlst, planlst, querymsg_d = out
                else: # pal, cot
                    slnlst, querymsg_lst = out
                # print(f"{len(set(slnlst))=}") # confirmed seed does not make the choices degenerate to one deterministic output
                    
                eval_pred_lst = [sln_eval(sln, row['ans'], method) for sln in slnlst]
                correct_sln_lst = [sln for (eval, pred), sln in zip(eval_pred_lst, slnlst) if eval]
                correct_pred_lst = [pred for (eval, pred), sln in zip(eval_pred_lst, slnlst) if eval]
                if not correct_sln_lst:
                    print("no correct solution found!")
                retries = dict(
                    method = method,
                    correct_solution = list(set(correct_sln_lst)),
                    correct_prediction = set(correct_pred_lst).pop() if correct_pred_lst else None,
                    backbone = backbone,
                    llm_kwargs = kwargs,
                )   
                if method == 'p2c': # for p2c, don't forget to cap the plan string to the code
                    # plan = planlst[0] if planlst else None
                    plan = planlst.pop()
                    correct_sln_lst = [f"{plan}\n{sln}".strip() for sln in correct_sln_lst]
                    retries['correct_solution'] = correct_sln_lst
                # embed the retries into the row 
                if 'retries' in row.keys():
                    row['retries'].append(retries)
                else:
                    row['retries'] = [retries]
                # except Exception as e:
                #     print(e)
            results.append(row)
        # save the results
        with jsl.open(outjslf, 'w') as writer:
            writer.write_all(results)
            print(f"saved {len(results)} records to {outjslf}")
        
            
            
            
            

            
if __name__ == "__main__":
    Fire(main)