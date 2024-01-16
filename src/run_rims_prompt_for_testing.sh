## this script is for understanding how things and codes work, but the actual experiment will run on the same python scripts.
# gsm_jslf is for dataset of the same format as gsm8K_test.jsonl. It could be other (svamp, single_op,... )



# 1 model-selection-resoning baseline
python run_inference.py  baseline_inference \
                --backbone chatgpt \
                --gsm_jslf ../../dataset/gsm8K_test.jsonl

# 2 rims algorithm run the 1's result
python run_inference.py  rims_inference  \
                --backbone  chatgpt  \
                --gsm_jslf ../../dataset/{result_file_from_above}.jsonl \
                --prompt_f  prompt_construction_src/prep_rims_prompts/gsm_prompts/3_reflectonce_cot2p2c.pal2cot.pal2p2c.txt_rm_ans  

# 3 evaluate 
python run_evaluation.py --jslf {result you want to eval}.jsonl