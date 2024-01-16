# rims_minimal
이전 코드베이스가 엉망인 관계로 제공함
시작점: https://github.com/fgenie/Model-Selection-Reasoning/tree/59debd8441e7cb4b7d733f256b20870073e36c08

프로젝트 구조가 좀 바뀌어서 수정중/디버깅 요함 (수)

## todo
 - [ ] fix bugs from revising project structure
 - add 
    - [ ] erasing (w/o affecting the result of the prev) selection result of "conflict rows" to avoid contamination
    - [ ] evaluation minimal script into `experiment_example.sh`
    - [ ] ocw symbolic answer evaluation/normalization code
    - [ ] ocw, (math) symbolic including prompts

## What to do before run
1. place `openai_key.txt` into `utils/`
2. take a look at `experiment_example.sh`. It does...
    - 1: run model-selection-reasoning prompt on {dataset}.jsonl
    - 2: run one of gsm rims prompt on {result of 1}.jsonl
        - (under construction) rims prompt will be queried only if the answer already inferred in 1 is in conflict. (conflict only run)
    - 3: you can see the evaluation result for one jsonl file from the last commandline of the script.   
