import numpy as np
from pathlib import Path 
import jsonlines as jsl
import pandas as pd 
from fire import Fire


def make_summary(table:pd.DataFrame, skip_threshold:int=0)->pd.DataFrame:
    nonskip_mask = table.skipped <= skip_threshold
     
    abl_mask = table.name.apply(lambda s: 'ablat' in s)
    abl_mask &= nonskip_mask
    if (~abl_mask).sum()>0:
        nonabl = table.iloc[:, 1:][~abl_mask].sum()/(~abl_mask).sum()
    if (abl_mask).sum()>0:
        abl = table.iloc[:, 1:][abl_mask].sum()/(abl_mask).sum()
    # print(nonabl)
    # print(abl)
    nonabl = nonabl.apply(lambda x: np.round(x, 2))
    if (abl_mask).sum()>0:
        abl = abl.apply(lambda x: np.round(x, 2))
    if abl_mask.sum() >0:
        res = pd.DataFrame([nonabl, abl], index = ['rims', 'rims + ablation'])
    else:
        res = pd.DataFrame([nonabl], index = ['rims'])
    return res
    
def get_nonconf_corrects_total(baseline_records):
    df = pd.DataFrame(baseline_records)
    corrects_mask = (df.answer - df.majority_ans).abs() < 1e-3
    nonconf_mask = df.selection_or_rims.apply(lambda d: 'majority_vote' in d and d['majority_vote']) 
    df_nonconf = df[nonconf_mask]
    total = len(df_nonconf)
    nc_corrects = (corrects_mask & nonconf_mask).sum() 
    return np.array([nc_corrects, total])

def simplify_name(name):
    name = name.split("conflictonly__")[-1]    
    backbone, name = name.split("_merged_model_selection3_")
    isablate = 'ablate' in name
    name = name.split('_reflectonce_')[-1]
    return f"{backbone}: {'ablation + ' if isablate else ''}{name}"


def main(backbone='chatgpt',
         root='/Users/seonils/dev/llm-reasoners/examples/Model-Selection-Reasoning/dataset/gsm8K_test_model_selection_baseline/',
         tab_name='jan14_results_table.md',
         dataset_path = '../../dataset/gsm8K_test.jsonl',
         skip_threshold=0,
         ):
    tab_name = tab_name.replace('.md', f"_{Path(dataset_path).name}_{backbone}_skipthld{skip_threshold}.md")
    PTN=f'conflictonly__{backbone}_*/{backbone}_*.jsonl'
    baseline_f =list(Path(root).glob(f'{backbone}_*_model_selection3.jsonl'))
    baseline_records = []
    for f in baseline_f:
        baseline_records.extend(list(jsl.open(f)))

    NONCONF = get_nonconf_corrects_total(baseline_records)# np.array((1218, 1246)) # num corrects, num total 

    # directories to score the num_corrects over conflict only examples
    dirs = list(Path(root).glob(PTN))

    def count_corrects(record):
        df = pd.DataFrame(record)
        # reflect / nonreflect
        # error_mask = df.selection_or_rims.apply(lambda d: 'error' in d and d['error']) 
        error_mask = df.majority_ans.isna()
        error_mask |= df.selection_or_rims.apply(lambda d: 'error' in d and d['error'])

        reflect_mask = df.selection_or_rims.apply(lambda d: d['did_reflect']>0 if 'did_reflect' in d else False)
        nonreflect_mask = df.selection_or_rims.apply(lambda d: d['did_reflect']==0 if 'did_reflect' in d else False)
        num_error = error_mask.sum()
        df_reflect = df [ reflect_mask & (~error_mask) ] 
        df_nonrefl = df [ nonreflect_mask & (~error_mask) ] 
        
        # count_corrects
        correct_refl = ((df_reflect.answer-df_reflect.majority_ans).abs() < 1e-3).sum(), len(df_reflect)
        correct_nonrefl = ((df_nonrefl.answer-df_reflect.majority_ans).abs() < 1e-3).sum(), len(df_nonrefl)

        c_refl, c_nonrefl = map(np.array, [correct_refl, correct_nonrefl])
        
        return c_refl, c_nonrefl, num_error

    def count_corrects_bs(record):
        df = pd.DataFrame(record)
        corrects = (df.answer - df.majority_ans).abs() < 1e-3
        ncorrects = (corrects.sum(), len(corrects))
        num_failed = df.majority_ans.isna().sum()
        return np.array(ncorrects), num_failed

    def array2readable(array, percent=False):
        correct, total = array 
        return f"{correct}/{total} ({100*(correct/total):.2f}\%)"



    TOTAL = len(list(jsl.open(dataset_path))) 
    conf2records = {dir.parent.name: (rec:=list(jsl.open(dir)), TOTAL-len(rec)-NONCONF[1])  for dir in dirs}
    to_table = []
    for prompt, (record, skipped) in conf2records.items():
        if 'baseline' not in prompt:
            reflect, nonreflect, num_failed= count_corrects(record)
            # reflect, nonreflect = np.array([correct:int, total:int])
            # justfailure = int 
            to_table.append( {'name': prompt, 'total': NONCONF+reflect+nonreflect + np.array([0, num_failed]), 'conflict_only': reflect + nonreflect, 'reflect': reflect, 'nonreflect': nonreflect, 'nonconf': NONCONF, 'justfailed': num_failed, 'skipped': skipped} )
        # else: # baseline 
        #     corrects = count_corrects_bs(record)
        #     baseline_table.append( {'name': prompt, 'total': NONCONF+corrects, 'conflict_only': corrects} )
    baseline_records = list(jsl.open(baseline_f))
    bscorrects, num_failed = count_corrects_bs(baseline_records)
    baseline_conflict_only_corrects_total = bscorrects - NONCONF
    baseline_table = [
        {'name': 'baseline', 'total': NONCONF+baseline_conflict_only_corrects_total, 'conflict_only': baseline_conflict_only_corrects_total, 'nonconf': NONCONF, 'justfailed': num_failed, 'skipped': TOTAL-len(baseline_records)}
    ]
            

    table = pd.DataFrame(to_table)
    summary_table = make_summary(table, skip_threshold=skip_threshold)
    bstable = pd.DataFrame(baseline_table)
    table.name = table.name.apply(simplify_name)
    # bstable.name
    for col in ['total', 'conflict_only', 'nonconf', 'reflect', 'nonreflect']:
        if col in table:
            table[col] = table[col].apply(array2readable if col!= 'total' else lambda x: array2readable(x, percent=True))
        if col in bstable:
            bstable[col] = bstable[col].apply(array2readable if col!= 'total' else lambda x: array2readable(x, percent=True))
        if col in summary_table:
            summary_table[col] = summary_table[col].apply(array2readable if col!= 'total' else lambda x: array2readable(x, percent=True))  

    table = table.sort_values(by='name')
    with open(tab_name, 'w') as f:
        print('### baseline', file=f)
        print(bstable.to_markdown(), file=f)
        
        print(f'\n### summary ({skip_threshold=})', file=f)
        print(summary_table.to_markdown(), file=f)

        print('\n### in detail', file=f)
        print(table.to_markdown(), file=f)
    print(f'wrote to \n\t{tab_name}')


if __name__ == '__main__':
    Fire(main)