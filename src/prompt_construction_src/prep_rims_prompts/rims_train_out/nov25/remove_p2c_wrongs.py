import jsonlines as jsl
import pandas as pd 
from pathlib import Path 

to_fix = list(Path().glob("resolved_*.jsonl"))
parent = to_fix[0].parent
out_fs = [parent/f.name.replace('resolved', 'filtered_resolved') for f in to_fix]

def has_plan(p2c_generation:str):
    return p2c_generation.strip().startswith('1.')

for f, of in zip(to_fix, out_fs):
    records = list(jsl.open(f))   
    if f.name.startswith('resolved_p2c'):
        # inspect_failed_method
        df = pd.DataFrame(records)
        tokeep_mask = df['fail_solutions'].apply(lambda lst: has_plan(lst[0]))
        df = df[tokeep_mask]
        records = df.to_dict(orient='records')
    else:
        # inspect attempting agains
        for row in records:
            row['retries'] = [rt for rt in row['retries'] if rt['correct_solution'] and ((rt['method']!='p2c') or (rt['method'] == 'p2c' and has_plan(rt['correct_solution'][0]))) ] #editing retries

    with jsl.open(of, 'w') as writer:
        writer.write_all(records)
        print(of)
    
    