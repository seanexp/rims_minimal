import re
import yaml
from string import Template 
from pathlib import Path 
import jsonlines as jsl
import pandas as pd
from fire import Fire
from tqdm import tqdm
from collections import defaultdict
import json
# from ..tool import parse_python_code_from_string


import openai
openai.api_key = open('../../openai_key.txt').read().strip()
ABB2FULL = {'pal': 'Program-aided Language Modeling', 'cot': "Chain-of-Thought", "p2c": "Plan-and-then-Code"}


def parse_python_code_from_string(unparsed_txt: str):
    ptn = r"```python((.|\n)*?)```"
    match = re.search(ptn, unparsed_txt)
    if match is not None:
        return match.group(1)
    else:
        return None


# @retry(wait=wait_chain(*[wait_fixed(3) for i in range(5)])) #defining backoff for retrying.
def query_llm(msgs:list, 
              backbone:str='gpt4turbo', 
              stop:str=None,
              
              T:float=1.0, 
              seed:int=777,
              max_tokens:int=2048, 
              n:int=1, 
              ):
    if backbone == 'gpt4turbo':
        model = 'gpt-4-1106-preview'
    elif backbone == 'gpt4':
        model = 'gpt-4'
    elif backbone == 'chatgpt':
        model = 'gpt-3.5-turbo'
    else:
        raise ValueError(f"backbone {backbone} not supported")
    
    resp = openai.ChatCompletion.create(
                        messages=msgs,
                        model=model,
                        stop=stop,
                        
                        temperature=T,
                        seed=seed,
                        max_tokens=max_tokens,
                        n=n, 
                        )
    content = resp['choices'][0]['message']['content']
    return content

def flatten_retries(row):
    for retry_d in row['retries']:
        retrymethod = retry_d['method']
        retry_d = {f"retry_{k}_{retrymethod}":v for k,v in retry_d.items()}
        row.update(retry_d)
    return row

def replace_w_gen(userprompt:str, generation:str)->str:
    # postprocess generation
    genlines = [l for l in generation.split('\n') if l.strip()]
    genlines_d = {l[:l.find(": ")+1]:l for l in genlines if l.startswith('`Mistakes`:') or l.startswith("`Hint for a better Method choice`:")}
    keys = genlines_d.keys()
    # iterate over userprompt and replace
    promptlines = userprompt.split('\n')
    for i, line in enumerate(promptlines):
        for k in keys:
            if line.startswith(k):
                promptlines[i] = genlines_d[k]
    completed = "\n".join(promptlines)
    return completed


def remove_p2c_resolution_rows(retries:list):
    return [r for r in retries if r['method']!='p2c']
def remove_empty_solution_rows(retries:list):
    return [r for r in retries if r['correct_solution']]


def main(each_n=-1, nop2c:bool=False):
    yamld = yaml.full_load(open('2_prompt_generate_reflection_forceformat.yaml'))
    tmp = Template(yamld['user'])

    jslf_lst= list(Path('rims_train_out/nov25/').glob("*resolved_*gpt4turbo*.jsonl")) # resolved output contained 
    blurbd = defaultdict(list) # will store blurbd[failmethod2successmethod] = [blurb1, blurb2, ...]

    outf = f'2_blurb_{each_n}.json'


    for jslf in jslf_lst:
        if nop2c: 
            if 'p2c' in jslf.name:
                continue
        print(f"processing {jslf}")
        # filter jslf and pick 6 of interest with even length dist
        df = pd.DataFrame(jsl.open(jslf))
        df.fail_preds = df.fail_preds.apply(lambda x: [e for e in x if e is not None])
        df = df.query('fail_preds != []')
        print(df.shape)
        df.retries = df.retries.apply(remove_empty_solution_rows)
        if nop2c:
            df.retries = df.retries.apply(remove_p2c_resolution_rows)
        df = df [ df.retries.apply(len) > 0 ]
        print(df.shape)

        if each_n!=-1:
            indices = list(range(0, len(df), len(df)//each_n))[:each_n]
            df = df.iloc[indices]
        records = df.to_dict(orient='records')
        # reorder
        records = [flatten_retries(row) for row in records]

        for row in tqdm(records):
            # prep prompt
            for k in row.keys():
                try:
                    if k.startswith('retry_correct_solution_') and row[k]: 
                        save_k = f"{row['fail_method']}2{k.replace('retry_correct_solution_', '')}"
                        wrong_method = row['fail_method']
                        if wrong_method == 'pal':
                            for sln, wrong_pred in zip(row['fail_solutions'], row['fail_preds']):
                                if wrong_pred is not None:
                                    wrong_solution = parse_python_code_from_string(sln)
                                    break
                        else: 
                            wrong_pred = row['fail_preds'][0]
                            wrong_solution = row['fail_solutions'][0]
                        if wrong_pred is None or not wrong_solution:
                            continue
                        wrong_method = ABB2FULL[wrong_method] + f" ({wrong_method})"
                        ans = row['ans']
                        correct_method = k.replace('retry_correct_solution_', '')
                        correct_pred = row[f'retry_correct_prediction_{correct_method}'] # float or str
                        correct_solutions = row[k]
                                
                        if correct_method == 'pal': # this logic fails
                            for sln in correct_solutions:
                                correct_solution = parse_python_code_from_string(sln)
                                if correct_solution:
                                    break
                        else:
                            correct_solution = correct_solutions[0]
                        if not correct_solution:
                            continue
                        correct_method = ABB2FULL[correct_method] + f" ({correct_method})"
                        question = row['question']

                        

                        cp = correct_pred
                        wp = wrong_pred
                        # userprompt = tmp.substitute(QUESTION = question, WRONG_SOLUTION = wrong_solution, WRONG_PRED = wp, ANS = ans, CORRECT_SOLUTION = correct_solution, CORRECT_PRED = cp, WRONG_METHOD = wrong_method, CORRECT_METHOD = correct_method)
                        userprompt = tmp.substitute(QUESTION = question, WRONG_SOLUTION = wrong_solution, WRONG_PRED = wp, CORRECT_SOLUTION = correct_solution, CORRECT_PRED = cp, WRONG_METHOD = wrong_method, CORRECT_METHOD = correct_method) # do not include gt ans to the evaluating prompt it may make the model depend on it and cause problems in extending the blurbs

                        msgs = [
                            {'role':'system', 'content': yamld['system']},
                            {'role':'user', 'content': userprompt}
                        ]
                        generation = query_llm(msgs=msgs, backbone='gpt4turbo', stop=yamld['stop'], T=1.0, seed=777, max_tokens=2048, n=1)
                        completed = replace_w_gen(userprompt, generation)
                        print(completed)
                        print('=========')
                        print(generation)
                        blurbd[save_k].append(completed)
                        blurbd[f"{save_k}_genonly"].append(generation)
                except Exception as e:
                    print(e)
    json.dump(blurbd, open(outf, 'w'), indent=4)
    print(f"blurbs are written to {outf}")




if __name__ == '__main__':
    Fire(main)