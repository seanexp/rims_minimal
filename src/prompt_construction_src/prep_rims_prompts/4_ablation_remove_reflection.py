from pathlib import Path
import re

def remove_1st_attempt_and_edit_2nd(prompt:str)->str:
    # remove 1st attempt
    pattern = r"(`Method`:[\s\S]*?)`Workaround Method`"
    matches = re.findall(pattern, text, re.DOTALL)

    for txt in matches:
        prompt = prompt.replace(txt, "")
    assert len(matches) == 3, "there should be 3 matches"

    # edit 2nd attempts
    prompt = prompt.replace("`Workaround Method`", "`Method`")
    prompt = prompt.replace("`Answer 2`", "`Answer 1`")
    revised = prompt.replace("`Attempt 2`", "`Attempt 1`")
    
    return revised

reflect_once = list(Path().glob("3_reflectonce_*.txt")) 
for f in reflect_once:
    newf = f"4_ablate_{f.name}"
    text = f.open().read().strip()
    revised = remove_1st_attempt_and_edit_2nd(text)
    with open(newf, 'w') as wf:
        wf.write(revised)
        print(newf)