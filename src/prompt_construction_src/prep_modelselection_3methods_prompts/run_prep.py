import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.abspath("../../"))

import utils.llm_query_utils as llm_utils

if __name__ == "__main__":
    questions = []
    for line in math_prompt.TURBO_SELECT_USER.split("\n"):
        if line.lower().startswith("math problem: "):
            q = line.replace("Math problem: ", "").replace("Math Problem: ", "")
            questions.append(q)
            print(q)

    """
    Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
    Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
    Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
    Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
    Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
    """

    # for q in questions:
    #     codes, plans, querymsgs = query_plancode(
    #                                                 q,
    #                                                 plan_temperature=1.0,
    #                                                 code_temperature=1.0,
    #                                                 backbone='chatgpt',
    #                                                 n=1,
    #                                                 seed=777
    #                                                 )
    #     solution = plans[0] + "\n" + codes[0]
    #     print("====================================")
    #     print(f"Question: {q}")
    #     print(solution)
    p2c_chatgptout = open("p2c_chatgptout.txt").read().strip()
    q_sln_lst = p2c_chatgptout.split("====")

    ab_examples = math_prompt.TURBO_SELECT_USER.split(
        "Which of the above two choices can correctly answer the math problem?"
    )
    attempt_choice3 = (
        "Which of the above three choices can correctly answer the math problem?"
    )

    select3_prompt = []
    for q_sln, ab in zip(q_sln_lst, ab_examples):
        q, sln = q_sln.split("Question: ")[1].split("\n", 1)
        print(f"Question: {q}")
        print("solution")
        print(sln)
        print("answer")
        print(f"{safe_execute_turbo(sln)=}")
        print("====")

        abc_user = f"{ab}(C)\n{sln}\n\n{attempt_choice3}\n\n\n\n"
        select3_prompt.append(abc_user)

    select3_prompt = "".join(select3_prompt).strip()
    print(select3_prompt, file=open("select3user.txt", "w"))
