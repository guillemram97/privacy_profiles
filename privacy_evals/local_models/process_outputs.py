def process_evaluation(s: str):
    a_idx = s.find("[[A]]")
    b_idx = s.find("[[B]]")
    if a_idx != -1 and b_idx != -1:
        if a_idx < b_idx:
            s = "A"
        else:
            s = "B"
    elif a_idx != -1:
        s = "A"
    elif b_idx != -1:
        s = "B"
    else:
        s = "Undetected"
    return s


def aggregate_responses(s1: str, s2: str):
    if s1 == s2:
        return "C"
    return s1


def process_output(s: str, model_name: str):
    if model_name in ["gpt-4o", "gpt-4o-mini"]:
        return s
    header = return_header(model_name)
    footer = return_footer(model_name)
    if s is None or s == "":
        return ""
    s = s[s.find(header) + len(header) : s.rfind(footer)]
    return s


def return_header(model_name: str):
    if model_name.split("/")[0] == "meta-llama":
        return "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif model_name.split("/")[0] == "mistralai":
        return "[/INST] "
    elif 'gpt' in model_name:
        return "<|start|>assistant<|channel|>final<|message|>"


def return_footer(model_name: str):
    if model_name.split("/")[0] == "meta-llama":
        return "<|eot_id|>"
    elif model_name.split("/")[0] == "mistralai":
        return "</s>"
    elif 'gpt' in model_name:
        return "<|return|>"
