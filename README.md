# rims_minimal
이전 코드베이스가 엉망인 관계로 제공함
시작점: https://github.com/fgenie/Model-Selection-Reasoning/tree/59debd8441e7cb4b7d733f256b20870073e36c08


## What to do before run
place `openai_key.txt` into `utils/`

## How to Contribute

```
pip install pre-commit
pre-commit install
```

## How to run
### 1 model-selection-resoning baseline
```bash
python run_inference.py  baseline_inference \
                --backbone chatgpt \
                --outdir dbgoutdir/ \
                --gsm_jslf ../dataset/dbg.jsonl
```


### 2 rims algorithm run the 1's result (it will skip the non-conflict examples!)
```bash
python run_inference.py  rims_inference  \
                --backbone  chatgpt  \
                --outdir dbgoutdir/ \
                --prompt_f  prompt_construction_src/prep_rims_prompts/gsm_prompts/3_reflectonce_cot2p2c.pal2cot.pal2p2c.txt_rm_ans   \
                --gsm_jslf dbgoutdir/chatgpt_01_18_04_48_model_selection3_startidx0.jsonl
```

### 3 evaluate
```bash
# baseline result
python run_evaluation.py --eval_jslf dbgoutdir/chatgpt_01_18_04_48_model_selection3_startidx0.jsonl
# rims result
python run_evaluation.py --eval_jslf dbgoutdir/chatgpt_rims_01_18_04_49_startidx0.jsonl
```


## todo
 - [x] fix bugs from revising project structure
 - add
    - [x] erasing (w/o affecting the result of the prev) selection result of "conflict rows" to avoid contamination
    - [x] evaluation minimal script into `experiment_example.sh`
    - [ ] ocw symbolic answer evaluation/normalization code
    - [ ] ocw, (math) symbolic including prompts

## algorithm
TBA
