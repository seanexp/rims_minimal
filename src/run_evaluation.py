import jsonlines as jsl
import pandas as pd
from fire import Fire


def eval_gsm_svamp(df):
    return ((df.majority_ans - df.answer).abs() < 1e-3).sum()


def eval_math_ocw(df):
    raise NotImplementedError("to be implemented")


def main(eval_jslf: str):
    df = pd.DataFrame(jsl.open(eval_jslf))
    nonconflict_mask = df.selection_or_rims.apply(
        lambda d: d["majority_vote"] if "majority_vote" in d.keys() else False
    )
    fail_mask = df.selection_or_rims.apply(
        lambda d: d["error"] if "error" in d.keys() else False
    )  # api error
    conflict_mask = ~(nonconflict_mask | fail_mask)
    df_conflict_only = df[conflict_mask]
    df_nonconflict_only = df[nonconflict_mask]

    total = len(df)
    failcount = fail_mask.sum()

    nonconf_correct = eval_gsm_svamp(df_nonconflict_only)
    conf_correct = eval_gsm_svamp(df_conflict_only)
    print(f"total: {nonconf_correct + conf_correct} / {total}")
    print(
        f"fail: {failcount} / {total}",
    )
    print(f"nonconflict: {nonconf_correct} / {nonconflict_mask.sum()}")
    print(f"conflict: {conf_correct} / {conflict_mask.sum()}")


if __name__ == "__main__":
    Fire(main)
