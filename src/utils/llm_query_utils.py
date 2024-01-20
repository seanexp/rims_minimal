import re
from itertools import combinations
from pathlib import Path
from typing import Any

import func_timeout
import openai
import regex
import yaml

from . import math_prompt

# Get the absolute path of the current script
THIS_PARENT = Path(__file__).parent.resolve()

# Construct the path to the openai_key.txt file
key_file_path = THIS_PARENT / "openai_key.txt"

# Read the API key from the file
try:
    openai.api_key = open(key_file_path).read().strip()
except Exception as e:
    print(e)
    print(f"place your openai_key.txt inside utils/")


def exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(e)
            return None

    return wrapper


### almost same to string.Template, but with custom delimiter ( [QUESTION] == ${QUESTION}, to avoid `$` used frequently in price-related questions )
class PromptStr(str):
    def __init__(self, template_str):
        super().__init__()
        self += template_str

    def sub(self, placeholder_name: str, tobe: str):
        return PromptStr(self.replace(f"[{placeholder_name}]", str(tobe)))

    def sub_map(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.get_placeholder_names():
                self = self.sub(k, v)
        return self

    def get_placeholder_names(self) -> list:
        return re.findall(r"\[(.*?)\]", self)


### llm query functions ###
def query_cot(
    question: str, temperature: float = 0.0, backbone: str = "chatgpt", n=1, seed=777
):
    """
    This function is used to query OpenAI for CoT solutions.

    Args:
        data: a dict containing the question and answer
        key: the OpenAI API key
        temperature: the temperature used in CoT
        backbone: ChatGPT or GPT-4

    Returns:
        completions: a list containing the CoT solution
    """
    query_message = get_cot_prompt(question, backbone=backbone)
    # print(query_message)
    if backbone == "gpt4":
        model_name = "gpt-4"
    elif backbone == "gpt4turbo":
        model_name = "gpt-4-1106-preview"
    elif backbone == "chatgpt":
        model_name = "gpt-3.5-turbo"

    completions = []
    cot_solution = openai.ChatCompletion.create(
        # api_key=key,
        model=model_name,
        max_tokens=500,
        stop="\n\n\n",
        messages=query_message,
        temperature=temperature,
        top_p=1.0,
        seed=seed,
        n=n,
    )
    if n == 1:
        completions = [cot_solution["choices"][0]["message"]["content"]]
    else:
        completions = [
            cot_solution["choices"][i]["message"]["content"] for i in range(n)
        ]
    return completions, query_message


# actual llm query function for p2c method
def _query(  # key,
    model_name: str = "gpt-3.5-turbo",
    max_tokens: int = 2048,
    stop: str = None,
    messages=None,
    temperature=0.0,
    top_p=1.0,
    n=1,
    mode="plan",
    seed=777,
):  # mode = plan or code
    resp = openai.ChatCompletion.create(  # api_key=key,
        model=model_name,
        max_tokens=max_tokens,
        stop=stop,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        n=n,
        seed=seed,
    )
    if n == 1:
        content = resp["choices"][0]["message"]["content"]  # str
        if mode == "plan":
            plan = postprocess_plan(content)  # it will complain when failing
            return plan
        elif mode == "code":
            code = postprocess_code(content)
            return code
    else:  # n>1
        contents = [ch["message"]["content"] for ch in resp["choices"]]
        postprocess = postprocess_plan if mode == "plan" else postprocess_code
        res_strs = [postprocess(c) for c in contents]
        return res_strs


# p2c: querying plan and code separately inside
def query_plancode(
    question: str,  # data: dict,
    plan_temperature: float = 0.0,
    code_temperature: float = 0.0,
    backbone: str = "gpt-3.5-turbo",
    n=1,
    seed: int = 777,
):
    """
    PAL variant: 1. generate planning for the given question 2. based on 1, generate code like PAL does.

    args:
        mostly same arguments with `query_pal()` below
    returns:
        [list of codes], [list of plans (1)], {codequery: str, planquery: str}
    """
    # specify model
    if backbone == "gpt4":
        model_name = "gpt-4"
    elif backbone == "gpt4turbo":
        model_name = "gpt-4-1106-preview"
    elif backbone == "chatgpt":
        model_name = "gpt-3.5-turbo"

    if model_name.startswith("gpt-4"):
        # print(f'gpt-4 uses k_fewshot=5 as default (p2c fs_prompting)')
        k_fewshot = 5
    elif model_name.startswith("gpt-3.5-turbo"):
        # print(f'gpt-3.5 uses k_fewshot=8 as default (p2c fs-prompting)')
        k_fewshot = 8

    # generate plan (retry included)
    plan_query_msg = get_plan_prompt(question, k_fewshot=k_fewshot)
    # print(plan_query_msg)
    plan = _query(
        model_name=model_name,
        max_tokens=1024,
        stop="Question: ",
        messages=plan_query_msg,
        temperature=plan_temperature,
        top_p=1.0,
        n=1,
        mode="plan",
        seed=seed,
    )

    if plan:
        code_query_msg = get_plan2code_prompt(question, plan=plan, k_fewshot=k_fewshot)
        # print(code_query_msg)
        code = _query(
            model_name=model_name,
            max_tokens=1024,
            stop="Question: ",
            messages=code_query_msg,
            temperature=code_temperature,
            top_p=1.0,
            n=n,
            mode="code",
            seed=seed,
        )  # ,
        if not code:
            return (
                [None],
                [plan],
                {"codequery": code_query_msg, "planquery": plan_query_msg},
            )
        else:
            return (
                [code] if n == 1 else code,
                [plan],
                {"codequery": code_query_msg, "planquery": plan_query_msg},
            )
    else:
        return None, None, {"codequery": code_query_msg, "planquery": plan_query_msg}


def query_pal(question: str, temperature: float, backbone: str, n=1, seed=777):
    """
    This function is used to query OpenAI for PAL solutions.

    Args:
        data: a dict containing the question and answer
        key: the OpenAI API key
        temperature: the temperature used in PAL
        backbone: ChatGPT or GPT-4

    Returns:
        completions: a list containing the PAL solution
    """
    query_message = get_pal_prompt(question, backbone=backbone)
    # print(query_message)
    if backbone == "gpt4":
        model_name = "gpt-4"
    elif backbone == "gpt4turbo":
        model_name = "gpt-4-1106-preview"
    elif backbone == "chatgpt":
        model_name = "gpt-3.5-turbo"
    completions = []
    pal_solution = openai.ChatCompletion.create(
        model=model_name,
        max_tokens=500,
        stop="\n\n\n",
        messages=query_message,
        temperature=temperature,
        top_p=1.0,
        seed=777,
        n=n,
    )

    if n == 1:
        completions.extend(
            [choice["message"]["content"] for choice in pal_solution["choices"]]
        )  # wtf this code...
        completions = completions[:1]
    else:  # this line might not be compatible with self-consistency setting in the original code
        completions = [
            pal_solution["choices"][i]["message"]["content"] for i in range(n)
        ]
    return completions, query_message


def query_selection(
    question: str,
    backbone: str,
    cot_solution: str = "",
    pal_solution: str = "",
    p2c_plan_code_solution: str = "",
):
    def postprocess_selection(selection_str: str) -> str:
        ptn = r"\([A-C]\)"
        matches = re.findall(ptn, selection_str)
        choice = matches[0]

        choice2method = {"(A)": "cot", "(B)": "pal", "(C)": "p2c"}

        return choice2method[choice]

    if backbone == "gpt4":
        model_name = "gpt-4"
    elif backbone == "gpt4turbo":
        model_name = "gpt-4-1106-preview"
    elif backbone == "chatgpt":
        model_name = "gpt-3.5-turbo"

    cot_pal_p2c_solution_list = [cot_solution, pal_solution, p2c_plan_code_solution]
    cot_pal_p2c_solution_list = [
        s for s in cot_pal_p2c_solution_list if s
    ]  # remove p2c if empty

    cot_pal_p2c_solution_d = dict(zip("cot pal p2c".split(), cot_pal_p2c_solution_list))

    selection_message = get_select_prompt(
        question, cot_pal_p2c_solution_d, backbone=backbone
    )
    select_str = openai.ChatCompletion.create(
        model=model_name,
        max_tokens=200,
        seed=777,  # added on dec 21
        stop="\n\n",
        messages=selection_message,
        temperature=0.0,
        top_p=1.0,
        n=1,
    )["choices"][0]["message"]["content"]

    final_answer = postprocess_selection(select_str)
    return final_answer, select_str  # 'pal'|'p2c'|'cot'


def query_rims_inference(
    question: str,
    prompt_f: str,
    backbone: str,
    temperature: float = 0.0,
    n: int = 1,
    max_tokens: int = 2048,
    turn_based: bool = False,
    continue_writing_gpt_messages: list = None,  # list of messages to invoke continue writing down the rims prompt format.
    stop_tok=None,
    for_eval_or_extend: bool = False,
) -> tuple:
    #   modif_prompt:bool=True) -> tuple:
    if backbone == "chatgpt":
        model_name = "gpt-3.5-turbo-16k"
    elif backbone == "gpt4":
        model_name = "gpt-4"
    elif backbone == "gpt4turbo":
        model_name = "gpt-4-1106-preview"

    def get_turn_based_prompt(prompt_f: str, q: str = "") -> list:
        # n_fewshot:int=8)->list:
        prompt_f
        q

        raise NotImplementedError(
            "see 99_*yaml to implement here + query_enhanced_coh:get_turn_based_prompt()"
        )

    def parse_raw_modif(rawqueryout: str) -> dict:
        """
        helper for Attempt 1,2,3... variants

        1/ read prompt to detect what to parse (`Some string here` <-- to be parsed)
        2/ and then parse those into a dict
        """
        # read the output and get what to parse
        pattern = r"`(.*?)`:"
        to_parse = re.findall(pattern, rawqueryout)
        to_parse = list(set(to_parse) - {"Evaluation"})

        # read the output again to parse the designated fields
        parse_dd = dict()

        duplicated = 1

        for fld in to_parse:
            # pattern = rf"`{fld}`:\s*(.*?)(?=`|$)"
            # pattern = rf"`{fld}`:\s*(?:```)?(.*?)(?:```)?(?=`|$)"
            pattern = rf"`{fld}`:\s*(?:```)?([\s\S]*?)(?=(?:```)?\n`[A-Z]|$)"
            matches = re.findall(pattern, rawqueryout, re.DOTALL)
            if fld in {
                "Mistakes",
                "Hint for a better Method choice",
                "Workaround Method",
                "Method",
            }:  # Method supposed not to appear multiple times, for chatgpt, it happens, and maybe for other plms too.
                parse_dd[fld] = matches[::duplicated]
            else:
                duplicated = max(duplicated, len(matches))
                if len(matches) > 0:
                    parse_dd[fld] = matches[0].strip()
                else:
                    parse_dd[fld] = ""

        for (
            k
        ) in (
            parse_dd.keys()
        ):  # found erratic parsings of the rims code solutions (``` at the end not removed properly)
            if k.startswith("Attempt ") and parse_dd[k].strip().endswith("```"):
                parse_dd[k] = parse_dd[k].strip().rstrip("```").strip()

        return parse_dd

    def process_rims_out_dict(parse_dd: dict) -> dict:
        """
        in:
            parsed_dict: contains fields that is directly related to the prompt response such as...
                Attempt 1: solution1 string
                Answer 1: solution1 answer (raw string)
                Mistakes: [solution 1,2,3,...'s mistake string]
                ...
        out:
            eval_friendly_d (dict): contains eval-friendly parsed fields
                good_solution: solution string at the last
                good_ans: correct answer executed above
                good_method: correct method abbreviation (e.g. cot)
                bad_ans: [list of wrong answers]
                bad_method: [list of wrong methods before the correct one]
                bad_solutions: [list of wrong solutions before the correct one]
                mistakes: [list of mistakes]
                hint: [list of hints]

        """

        def get_answer_rims(solution: str, ans: str = "", method: str = ""):
            try:
                if method == "cot":
                    pred = parse_num_from_answer(ans)
                elif method == "pal":
                    pred = safe_execute_turbo(solution)
                elif method == "p2c":
                    code = separate_plan_code(solution)[
                        1
                    ]  # solution of p2c already processed by `postprocess_code()`
                    pred = safe_execute_turbo(code)
                else:
                    raise ValueError(
                        "method not in {cot, pal, p2c}, failed processing rims output ans"
                    )
            except Exception as e:
                print(e)
                pred = None
            return pred

        attempts_keys = sorted([k for k in parse_dd.keys() if "Attempt" in k])
        ans_keys = sorted([k for k in parse_dd.keys() if "Answer" in k])
        # method_keys = sorted([k for k in parse_dd.keys() if 'Method' in k])

        if (
            ans_keys and attempts_keys
        ):  # answer and solutions inside. additionally Method key is also in the parse_dd
            good_solution = parse_dd[attempts_keys[-1]] if attempts_keys else None
            did_reflect = 0
            if "Workaround Method" in parse_dd.keys():
                did_reflect += len(parse_dd["Workaround Method"])
                good_method = parse_method2(parse_dd["Workaround Method"][-1])
                bad_method = [parse_method2(parse_dd["Method"].pop())]
                if len(parse_dd["Workaround Method"]) > 1:
                    bad_method += [
                        parse_method2(mstr)
                        for mstr in parse_dd["Workaround Method"][:-1]
                    ]

                # ans and solutions
                good_ans = parse_dd[ans_keys[-1]]
                bad_ans = [parse_dd[ak] for ak in ans_keys[:-1]]

                good_solution = parse_dd[attempts_keys[-1]]
                bad_solution = [parse_dd[atk] for atk in attempts_keys[:-1]]

            elif "Method" in parse_dd.keys():
                if "Mistakes" in parse_dd.keys():
                    did_reflect += len(parse_dd["Mistakes"])
                good_method = parse_method2(parse_dd["Method"][-1])
                bad_method = [parse_method2(m) for m in parse_dd["Method"][:-1]]

                # ans and solutions
                good_ans = parse_dd[ans_keys[-1]]
                bad_ans = [parse_dd[ak] for ak in ans_keys[:-1]]

                good_solution = parse_dd[attempts_keys[-1]]
                bad_solution = [parse_dd[atk] for atk in attempts_keys[:-1]]

            else:  # solved at once
                good_method = parse_method2(parse_dd["Method"])
                bad_method = []

                good_ans = parse_dd[ans_keys[-1]]
                bad_ans = []

                good_solution = parse_dd[attempts_keys[-1]]
                bad_solution = []

        else:  # rims queried for evaluation only. no answer nor solutions.
            did_reflect = 0
            good_solution = None
            good_method = None
            good_ans = None
            bad_solution = []
            bad_ans = []
            bad_method = []

        mistakes = []
        hint = []
        if "Mistakes" in parse_dd.keys():
            mistakes = parse_dd["Mistakes"]
        if "Hint for a better Method choice" in parse_dd.keys():
            hint = parse_dd["Hint for a better Method choice"]

        if not len(bad_solution) == len(bad_ans) == len(bad_method):
            print(f"{bad_solution=}", f"{bad_ans=}", f"{bad_method=}")
            print(f"{good_solution=}", f"{good_ans=}", f"{good_method=}")
            raise ValueError(
                f"{bad_solution=} possibly repetition generated (chatgpt, temp 0)"
            )  # the row will be skipped (raised when generation has Attempt 1 after Attempt 1 or similar behaviors)

        eval_friendly_d = dict(
            good_solution=good_solution,
            good_ans=get_answer_rims(good_solution, ans=good_ans, method=good_method),
            good_method=good_method,
            bad_solutions=bad_solution,
            bad_ans=[
                get_answer_rims(s, ans=a, method=m)
                for s, a, m in zip(bad_solution, bad_ans, bad_method)
            ],
            bad_method=bad_method,
            mistakes=mistakes,
            hint=hint,
            did_reflect=did_reflect,
        )
        return eval_friendly_d

    if turn_based:  # *.yaml
        messages = get_turn_based_prompt(prompt_f, q=question, n_fewshot=n_fewshot)
    else:  # *.txt  # DEC4 exps
        rawprompt = open(prompt_f).read().strip()
        prompt_tmp = PromptStr(rawprompt)
        prompt = prompt_tmp.sub("QUESTION", question)  # data['question'])
        assert isinstance(prompt, str)
        messages = [{"role": "user", "content": prompt}]
        if continue_writing_gpt_messages is not None:
            assert isinstance(
                continue_writing_gpt_messages, list
            ), f"continue_writing_gpt_messages should be a list of messages to openai chat create {continue_writing_gpt_messages=}"
            messages.extend(continue_writing_gpt_messages)
    if stop_tok is None:  # decode until it faces correct answer
        stop_tok = [
            "\n`Evaluation`: Correct",
            "Evaluation: Correct",
        ]  # could be a list or a single string object. Defaults: None
    if n == 1:
        raw_query_out = openai.ChatCompletion.create(
            # api_key=key,
            seed=777,
            model=model_name,
            max_tokens=max_tokens,
            stop=stop_tok,
            messages=messages,
            temperature=temperature,
            n=n,
            # top_p=1.0,
        )["choices"][0]["message"][
            "content"
        ]  # str
        if continue_writing_gpt_messages is not None:
            msgs_except_inst = continue_writing_gpt_messages[:-1]
            if (
                msgs_except_inst
            ):  # left outputs to prepend (for the ease of postprocessing (...? maybe?) )
                given_as_msgs_str = "\n".join([m["content"] for m in msgs_except_inst])
                raw_query_out = given_as_msgs_str + "\n" + raw_query_out
                raw_query_out = raw_query_out.strip()
        parsed_dict = parse_raw_modif(raw_query_out)
        try:
            eval_friendly_d = process_rims_out_dict(parsed_dict)
        except:
            print(f"{raw_query_out=}")
            print(f"{parsed_dict=}")
            print()
            raise ValueError("failed processing rims output")

        return eval_friendly_d, parsed_dict, raw_query_out, messages

    else:  # later unify into the below format. --> Need to correct the code uses inside `src/portable/`
        raw_query_outs = [
            openai.ChatCompletion.create(
                # api_key=key,
                seed=777,
                model=model_name,
                max_tokens=1024,
                stop=stop_tok,
                messages=messages,
                temperature=temperature,
                n=n,
                # top_p=1.0,
            )["choices"][i]["message"]["content"]
            for i in range(n)
        ]  # str
        if continue_writing_gpt_messages is not None:
            msgs_except_inst = continue_writing_gpt_messages[:-1]
            if (
                msgs_except_inst
            ):  # left outputs to prepend (for the ease of postprocessing (...? maybe?) )
                given_as_msgs_str = "\n".join([m["content"] for m in msgs_except_inst])
                raw_query_outs = [
                    (given_as_msgs_str + "\n" + rqo).strip() for rqo in raw_query_outs
                ]
        parsed_dicts = [
            parse_raw_modif(raw_query_out) for raw_query_out in raw_query_outs
        ]
        eval_friendly_ds = [
            process_rims_out_dict(parsed_dict) for parsed_dict in parsed_dicts
        ]

        return eval_friendly_ds, parsed_dicts, raw_query_outs, messages


### getting prompts for each method ###
def get_select_prompt(
    question: str, cot_pal_p2c_sln_d: dict, backbone: str = "chatgpt"
):
    """
    This function is used to generate the selection prompt.
    """
    if len(cot_pal_p2c_sln_d) == 3:
        if backbone == "gpt4" or backbone == "gpt4turbo":
            system_message = math_prompt.GPT4_SELECT_SYSTEM3
            user_message = math_prompt.GPT4_SELECT_USER3
            assistant_message = math_prompt.GPT4_SELECT_ASSISTANT3
        elif backbone == "chatgpt":
            system_message = math_prompt.TURBO_SELECT_SYSTEM3
            user_message = math_prompt.TURBO_SELECT_USER3
            assistant_message = math_prompt.TURBO_SELECT_ASSISTANT3
    elif len(cot_pal_p2c_sln_d) == 2:
        if backbone == "gpt4" or backbone == "gpt4turbo":
            system_message = math_prompt.GPT4_SELECT_SYSTEM
            user_message = math_prompt.GPT4_SELECT_USER
            assistant_message = math_prompt.GPT4_SELECT_ASSISTANT
        elif backbone == "chatgpt":
            system_message = math_prompt.TURBO_SELECT_SYSTEM
            user_message = math_prompt.TURBO_SELECT_USER
            assistant_message = math_prompt.TURBO_SELECT_ASSISTANT
    else:
        assert (
            False
        ), f"len(cot_pal_p2c_sln_d) needs to be 2 or 3 (current = {len(cot_pal_p2c_sln_d)})"

    cot_solution, pal_solution, p2c_solution = cot_pal_p2c_sln_d.values()

    messages = get_user_assistant_messages(
        system_message, user_message, assistant_message
    )

    try:  # looks super unhappy, but keep this to maintain consistency of the code and results...
        pal_solution_lines_strip = [l.strip for l in pal_solution.split("\n")]
        docstring_idxs = [
            i
            for i, x in enumerate(pal_solution_lines_strip)
            if x == '"""' or x == "'''"
        ]
        dsstart, dsend = min(docstring_idxs), max(docstring_idxs)

        pallines = [l for l in pal_solution.split("\n")]
        pal_generated = "\n".join(pallines[:dsstart] + pallines[dsend + 1 :])
    except Exception as e:
        pal_generated = (
            pal_solution[0].strip()
            if isinstance(pal_solution, list)
            else pal_solution.strip()
        )

    if cot_solution[0].startswith(
        "Answer:"
    ):  # put 'Answer:' at the start of CoT answer generation. Original code does this but not sure what they really wanted to do with this... biasing toward CoT?
        cot_generated = (
            cot_solution[0].strip()
            if isinstance(cot_solution, list)
            else cot_solution.strip()
        )
    else:
        cot_generated = (
            "Answer:\n" + cot_solution[0].strip()
            if isinstance(cot_solution, list)
            else "Answer:\n" + cot_solution.strip()
        )

    if len(cot_pal_p2c_sln_d) == 2:
        user_message = f"""Math problem: {question.strip()}

(A)
{cot_generated.strip()}

(B)
{pal_generated.strip()}

Which of the above two choices can correctly answer the math problem?"""

    else:  # len(cot_pal_p2c_sln_d)==3:
        p2c_choice_str = f"(C)\n{p2c_solution[0].strip() if isinstance(p2c_solution, list) else p2c_solution.strip()}\n\nWhich of the above three choices can correctly answer the math problem?"
        user_message = user_message.replace(
            "Which of the above two choices can correctly answer the math problem?",
            p2c_choice_str,
        )

    messages += [{"role": "user", "content": user_message}]

    return messages


def get_user_assistant_messages(
    system_message: str, user_message: str, assistant_message: str
):
    """
    This function is used to convert the prompt into the message format used by OpenAI Chat API.
    """
    messages = []
    messages.append({"role": "system", "content": system_message})
    split_user_messages = user_message.split("\n" * 4)
    split_assistant_messages = assistant_message.split("\n" * 4)  # delim==4*\n...
    for i in range(
        len(split_user_messages)
    ):  # user messages and assistant messages are paired... actually. This should have been `zip()`.
        question = split_user_messages[i]
        answer = split_assistant_messages[i]
        messages += [
            {"role": "user", "content": f"{question}"},
            {"role": "assistant", "content": f"{answer}"},
        ]
    return messages


def get_cot_prompt(question: str, backbone: str):
    """
    This function is used to generate the CoT prompt.
    append "Question: " to the `question`
    """
    if backbone == "gpt4" or backbone == "gpt4turbo":
        system_message = math_prompt.GPT4_COT_SYSTEM
        user_message = math_prompt.GPT4_COT_USER
        assistant_message = math_prompt.GPT4_COT_ASSISTANT
    elif backbone == "chatgpt":
        system_message = math_prompt.TURBO_COT_SYSTEM
        user_message = math_prompt.TURBO_COT_USER
        assistant_message = math_prompt.TURBO_COT_ASSISTANT

    messages = get_user_assistant_messages(
        system_message, user_message, assistant_message
    )
    messages += [{"role": "user", "content": f"Question: {question}"}]

    return messages


def get_pal_prompt(question: str, backbone: str):
    """
    This function is used to generate the PAL prompt.
    """
    if backbone == "gpt4" or backbone == "gpt4turbo":
        system_message = math_prompt.GPT4_PAL_SYSTEM
        user_message = math_prompt.GPT4_PAL_USER
        assistant_message = math_prompt.GPT4_PAL_ASSISTANT
        messages = get_user_assistant_messages(
            system_message, user_message, assistant_message
        )

        messages += [
            {"role": "user", "content": f"Question: {question}\n\n# solution in Python"}
        ]

    elif backbone == "chatgpt":
        system_message = math_prompt.TURBO_PAL_SYSTEM
        user_message = math_prompt.TURBO_PAL_USER
        assistant_message = math_prompt.TURBO_PAL_ASSISTANT
        messages = get_user_assistant_messages(
            system_message, user_message, assistant_message
        )

        messages += [
            {
                "role": "user",
                "content": f"Answer the following question in Python: {question}",
            }
        ]
    return messages


def get_plan_prompt(question: str, k_fewshot: int = 0) -> str:
    """
    prep prompt for plan generation
    put "Question: " in front of the `question`

    """
    PLAN_F = THIS_PARENT / "prompts_plan_v2.yaml"
    PLAN_PROMPTS_D = yaml.full_load(open(PLAN_F))
    prompt_d = PLAN_PROMPTS_D

    # q = data['question']
    q = question
    system = prompt_d["system_msg"]
    user_tmp = prompt_d["user_template"]
    user_attempt = user_tmp.replace("{QUESTION}", f"Question: {q}")

    fewshots_user = prompt_d["fewshots_user"][
        :k_fewshot
    ]  # list of fewshot strings include Question: as a stop sequence.
    fewshots_assistant = prompt_d["fewshots_assistant"][:k_fewshot]

    msgs = [
        {"role": "system", "content": system},
    ]
    for fu, fa in zip(fewshots_user, fewshots_assistant):
        usr = {"role": "user", "content": fu}
        astnt = {"role": "assistant", "content": fa}
        msgs.append(usr)
        msgs.append(astnt)
    msgs.append({"role": "user", "content": user_attempt})

    return msgs


def get_plan2code_prompt(
    question: str,  # data:dict,
    plan: str = "",
    k_fewshot: int = 0,
    custom_idxs: list = None,
):
    # little bit revision from PAL prompt.
    # `solution()` is returned (can execute with solution() call w/o argument
    """
    prep prompt for plan generation
    put "Qu
    estion: " in front of the `question`
    """
    CODE_F = THIS_PARENT / "prompts_code_v2.yaml"
    prompt_d = yaml.full_load(open(CODE_F))

    q = question  # data['question']
    system = prompt_d["system_msg"]
    user_tmp = prompt_d["user_template"]
    user_attempt = user_tmp.replace("{PLAN}", plan).replace(
        "{QUESTION}", f"Question: {q}"
    )

    if not custom_idxs:
        fewshots_user = prompt_d["fewshots_user"][
            :k_fewshot
        ]  # list of fewshot strings include Question: as a stop sequence.
        fewshots_assistant = prompt_d["fewshots_assistant"][:k_fewshot]
    else:
        fewshots_user = [prompt_d["fewshots_user"][i] for i in custom_idxs]
        fewshots_assistant = [prompt_d["fewshots_assistant"][i] for i in custom_idxs]

    msgs = [
        {"role": "system", "content": system},
    ]
    for fu, fa in zip(fewshots_user, fewshots_assistant):
        usr = {"role": "user", "content": fu}
        astnt = {"role": "assistant", "content": fa}
        msgs.append(usr)
        msgs.append(astnt)
    msgs.append({"role": "user", "content": user_attempt})

    return msgs


### postprocessing helpers ###
# for p2c response
def postprocess_plan(rawanswer: str):
    # lines = [l for l in rawanswer.split('\n') if '</end>' not in l]
    lines = rawanswer.split("\n")
    if len(lines) >= 1:
        plan_ = "\n".join(lines)
    else:
        print("plan gen failed")
        print(f"{rawanswer=}")
        plan_ = ""
    return plan_


### python code parsing for p2c... I know it's non-equivalent to the other function here. Used only for p2c
def postprocess_code(rawanswer: str, k_fewshot: int = 0):
    def remove_prints(code: str) -> str:
        lines = code.split("\n")
        lines_ = [
            l if not l.startswith("print(") else l.replace("print(", "# print(")
            for l in lines
        ]
        code_ = "\n".join(lines_)
        return code_

    try:
        # 1 removing starting wrap ```
        if "```python" in rawanswer:
            code = rawanswer.split("```python")[-1]
        elif rawanswer.startswith("```"):
            rawanswer = rawanswer.split("```")[-1]

        # 2 removing ``` at the end
        code = rawanswer.split("```")[0]  # ending ``` removal

        code = remove_prints(code)
        assert code
    except:
        print("code gen fails (unexecutable or funcname?)")
        print(f"code:\n{rawanswer}")
        code = ""
    return code


# p2c response postprocessing utility
def separate_plan_code(rawstr: str) -> tuple:
    # used for 5_cohlike_prompt
    # p2c results in plan\ncode so split it.
    rawstr = rawstr.strip()
    lines = rawstr.split("\n")
    found_code = False
    for i, l in enumerate(lines):
        if l.startswith("def ") and l.strip().endswith(":"):
            found_code = True
            break
    if found_code:
        plan = "\n".join(lines[:i])
        code = "\n".join(lines[i:])
    else:
        plan, code = None, None
    return plan, code


# method name normalization for rimsprompt
def parse_method2(methodstr: str) -> str:
    # works for --rimsprompt option
    normalized = methodstr.replace("-", " ").replace("_", " ").lower()
    norm2short = {
        "chain of thought": "cot",
        "cot": "cot",
        "program aided language modeling": "pal",
        "program aided language model": "pal",
        "pal": "pal",
        "plan and then code": "p2c",
        "p2c": "p2c",
    }  # this should be key as abb, and value as a set of component patterns for capturing
    for k in norm2short.keys():
        if k in normalized:
            return norm2short[k]
    else:
        return methodstr


# rims prompt: cot answer extracting postprocessing
def parse_num_from_answer(rawstr) -> float:
    """
    used for parsing number out from Answer (dec 4 exp)
    """
    rawstr = rawstr.replace(",", "")
    ptn = r"(-?\d+\.\d+|\d+)"
    nums = re.findall(ptn, rawstr)
    if not nums:
        return None
    else:  # more than one number
        return float(nums[-1])


### postprocess backticks for pythoncode output
def parse_python_code_from_string(unparsed_txt: str):
    ptn = r"```python((.|\n)*?)```"
    match = re.search(ptn, unparsed_txt)
    if match is not None:
        return match.group(1)
    else:
        return None


def get_func_name_from_string(codestring: str) -> str:
    match = re.search(r"def (\w+)\(", codestring)
    if match:
        funcname = match.group(1)
    return funcname


def _execute(code, code_return: str):
    # these imports are for locals() (to provide `global() context` to exec()
    import itertools
    import math
    import random
    from fractions import Fraction

    import sympy
    import sympy as sp
    from sympy import Symbol
    from sympy import isprime as is_prime
    from sympy import symbols

    try:
        locals_ = locals()
        solution = locals_.get("solution", None)
        funcname = get_func_name_from_string(code)  # for nontrivial function names

        if solution is not None:
            return solution()
        elif funcname:  # if any function name appears
            new_code = "import math\n" + code + f"\nresult = {funcname}()"
            loc = {}
            exec(new_code, locals(), loc)

            result = loc["result"]
            return result
        else:
            executed_code = (
                "import math\n"
                + "import datetime\n"
                + "\n".join([xx[4:] for xx in code.strip().split("\n")[1:-1]])
            )
            exec(executed_code, {}, locals())
            locals_ = locals()
            return locals_.get(code_return, None)

    except Exception as exp:
        print("Executing code error", exp)
        return None


### executing a code
def safe_execute_turbo(code_string: str):
    # === find code snippets between def solution(): and return ===
    try:
        code_list = code_string.strip().split("\n")

        new_code_list = []
        all_codes = []
        code_return = "ans"

        for i in range(len(code_list)):
            # if code_list[i].strip() == 'def solution():':
            if re.search(r"def (\w+)\(", code_list[i]) and code_list[i].startswith(
                "def "
            ):  # avoid including inner function definition
                new_code_list.append(code_list[i])
                for j in range(i + 1, len(code_list)):
                    if code_list[j].startswith("    "):
                        new_code_list.append(code_list[j])
                    if code_list[j].startswith(
                        "    return "
                    ):  # affirms outtermost return
                        code_return = code_list[j].split("return ")[1].strip()
                        break  # it could possibly miss the return if the function written with if-elif-else return at the end, which might be scarce.
                all_codes.append("\n".join(new_code_list))
                new_code_list = []

        if all_codes:
            new_code = all_codes[-1]

            ans = func_timeout.func_timeout(
                3,
                _execute,
                args=(
                    new_code,
                    code_return,
                ),
            )
        else:
            ans = None
    except (func_timeout.FunctionTimedOut, IndexError):
        ans = None

    return ans


# parsing (executing) cot result into a float
def extract_num_turbo(solution: str):
    ans = solution.strip().split("\n")[-1].replace("So the answer is ", "")
    prd = [x[0] for x in regex.finditer(r"[\d\.,]+", ans) if regex.search(r"\d", x[0])]
    if len(prd) > 2:
        prd = prd[-1]
    elif len(prd):
        prd = prd[0]
    else:
        prd = None
    try:
        prd = float(prd.replace(",", "").rstrip(".")) if prd else prd
    except:
        prd = None
    return prd


# ### retry wrapper ###
# @retry(wait=wait_chain(*[wait_fixed(3) for i in range(5)])) #defining backoff for retrying.
# def do_with_tenacity(func, *args, **kwargs):
#     return func(*args, **kwargs)


def get_concordant_answer(answers: list, ensure_unanimity: bool = False):
    """
    check if there is a pair of concordant answers.
    input: cot_ans, pal_ans, p2c_ans, [, ...]
    output: ans if concordant else None

    *recommend to put answers in the order of cot going first (usually they are intgers)
    """

    answers_no_none = [a for a in answers if a is not None]
    if ensure_unanimity:
        if len(set(answers_no_none)) == 1:
            return answers_no_none.pop()
        else:
            return None

    if len(answers_no_none) == 0:
        return None
    elif len(answers_no_none) == 1:
        return answers_no_none.pop()
    elif len(answers_no_none) == 2:
        if abs(answers_no_none[0] - answers_no_none[1]) < 1e-3:
            return answers_no_none[0]
        else:
            return None
    else:  # >=3
        for a1, a2 in combinations(answers_no_none, 2):
            if abs(a1 - a2) < 1e-3:
                return a1
        return None  # no concordant answers


def solution2blurb(method: str = "", solution: str = "", ans: Any = ""):
    """
    This function is for `--eval_indiv_method` option in `rims_inference` function
    solution into blurb string
    """
    abbr2full = {
        "cot": "Chain-of-Thought",
        "pal": "Program-aided Language Modeling",
        "p2c": "Plan-and-then-Code",
    }
    blurb_str = f"`Method`: {abbr2full[method]} ({method})\n`Attempt 1`: {solution}\n`Answer 1`: {ans}"
    return blurb_str


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
