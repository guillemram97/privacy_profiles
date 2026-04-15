import pandas as pd
import os
import json
import re
import pdb


def load_data(
    num_datapoints: int,
    num_splits: int,
    n_init: int,
    agent_type: str,
    id_experiment: str,
    outputs_dir: str,
    model_name: str,
    persona: str = None,
    model_answer_post: str = None,
    config: list[dict] = None,
    evaluated1: dict = None,
    evaluated2: dict = None,
) -> pd.DataFrame:
    if persona == "basic":
        data = pd.read_csv("data/data.csv", nrows=num_datapoints)
    else:
        data = pd.read_csv(f"data/data_{persona}.csv", nrows=num_datapoints)

    id_tmp, response_tmp, _ = load_response(
        outputs_dir, model_name, agent_type, id_experiment
    )
    if id_tmp != -1:
        to_drop = []
        for idx in range(len(data)):
            aux = int(data.iloc[idx]["idx"])
            if aux in id_tmp:
                to_drop.append(idx)
        data = data.drop(to_drop).reset_index(drop=True)    
    if num_splits > 1:
        start = int((len(data) / num_splits) * n_init)
        end = int((len(data) / num_splits) * (n_init + 1))
        if n_init == num_splits - 1:
            data = data.iloc[start:].reset_index(drop=True)
        else:
            data = data.iloc[start:end].reset_index(drop=True)
    if agent_type == "answer" and id_experiment == "-1":
        return data
    if agent_type == "paraphraser" or agent_type == "deferral":
        return data
    if agent_type == "answer_post" or agent_type == "deferral_post":
        to_drop = []
        agent_type_tmp = "paraphraser"
        if id_experiment == 'public':
            model_name_tmp = model_name
        else:
            model_name_tmp = [
                x["model_name"] for x in config if x["agent_type"] == agent_type_tmp
            ][0]
        id_tmp, response_tmp, _ = load_response(
            outputs_dir, model_name_tmp, agent_type_tmp, id_experiment
        )
        for idx in range(len(data)):
            aux = data.iloc[idx]
            if int(aux['idx']) in id_tmp:
                tmp_response = response_tmp[list(id_tmp).index(int(aux['idx']))]
                if  "[[[ ### createdPrompt ### ]]]" in tmp_response:
                    tmp_response = tmp_response[tmp_response.find("[[[ ### createdPrompt ### ]]]") + len("[[[ ### createdPrompt ### ]]]"):]
                    tmp_response = tmp_response[:tmp_response.find("[[[ ### completed ### ]]]")]
                    data.at[idx, "query_modified"] = tmp_response
                elif id_experiment == 'public':
                    data.at[idx, "query_modified"] = tmp_response
                else: 
                    to_drop.append(idx)
            else:
                to_drop.append(idx)
        data = data.drop(to_drop).reset_index(drop=True)
        return data
    if agent_type == "aggregator":
        if id_experiment == "-1":
            agent_type_tmp = "answer"
        else:
            agent_type_tmp = "answer_post"
        model_name_tmp = model_answer_post
        id_tmp, response_tmp, prompts_tmp = load_response(
            outputs_dir, model_name_tmp, agent_type_tmp, id_experiment
        )
        to_drop = []
        for idx in range(len(data)):
            aux = data.iloc[idx]
            if int(aux['idx']) in id_tmp or str(aux['idx']) in id_tmp:
                data.at[idx, "response"] = response_tmp[list(id_tmp).index(int(aux['idx']))]
                data.at[idx, "query_modified"] = prompts_tmp[list(id_tmp).index(int(aux['idx']))]
            else:
                to_drop.append(idx)
        data = data.drop(to_drop).reset_index(drop=True)  
    if agent_type == "evaluator":
        id_tmp_A, response_A = load_response(
            outputs_dir,
            evaluated1["model_name"],
            evaluated1["agent_type"],
            evaluated1["id_experiment"],
        )
        id_tmp_B, response_B = load_response(
            outputs_dir,
            evaluated2["model_name"],
            evaluated2["agent_type"],
            evaluated2["id_experiment"],
        )

        for idx in range(len(data)):
            aux = data.iloc[idx]
            if int(aux['idx']) in id_tmp_A:
                data.at[idx, "response_A"] = response_A[list(id_tmp_A).index(int(aux['idx']))]
            if int(aux['idx']) in id_tmp_B:
                data.at[idx, "response_B"] = response_B[list(id_tmp_B).index(int(aux['idx']))]
    return data



def load_response(
    outputs_dir: str, model_name: str, agent_type: str, id_experiment: str
):
    try:
        f = open(f"{outputs_dir}/{agent_type}/{model_name}/{id_experiment}.json", "r")
        lines = f.readlines()
        ids_tmp = []
        response_tmp = []
        prompts_tmp = []
        
        for line in lines:
            try:
                tmp_point = json.loads(line)
                if agent_type == "paraphraser":
                    if "response" in tmp_point and "[[[ ### createdPrompt ### ]]]" in tmp_point["response"]:
                        ids_tmp.append(tmp_point["idx"])
                        response_tmp.append(tmp_point["response"])
                        if "prompt" in tmp_point:
                            prompts_tmp.append(tmp_point["prompt"])
                    else:
                        ids_tmp.append(tmp_point["idx"])
                        if "response" in tmp_point:
                            response_tmp.append(tmp_point["response"])
                        elif "redacted_text" in tmp_point:
                            response_tmp.append(tmp_point["redacted_text"])
                        if "prompt" in tmp_point:
                            prompts_tmp.append(tmp_point["prompt"])
                elif agent_type == "deferral" or agent_type == "deferral_post":
                    if "[[[ ### label ### ]]]" in tmp_point["response"]:
                        ids_tmp.append(tmp_point["idx"])
                        tmp_point["response"] = tmp_point["response"][tmp_point["response"].find("[[[ ### label ### ]]]") + len("[[[ ### label ### ]]]"):]
                        if "[[[ ### completed ### ]]]" in tmp_point["response"]:
                            tmp_point["response"] = tmp_point["response"][:tmp_point["response"].find("[[[ ### completed ### ]]]")]
                        response_tmp.append(tmp_point["response"])
                else:
                    ids_tmp.append(tmp_point["idx"])
                    response_tmp.append(tmp_point["response"])
                    if agent_type == "answer_post":
                        if "prompt" in tmp_point:
                            prompts_tmp.append(tmp_point["prompt"])
            except:
                continue  # Skip the line if it can't be parsed or is missing keys
        f.close()
        return ids_tmp, response_tmp, prompts_tmp
    except:
        return -1, -1, -1


def save_output(
    outputs_dir: str,
    idx: int,
    response: str,
    prompt: str,
    model_name: str,
    agent_type: str,
    id_experiment: str,
    price: float = 0,
) -> None:
    if not os.path.exists(f"{outputs_dir}/{agent_type}/{model_name}"):
        os.makedirs(f"{outputs_dir}/{agent_type}/{model_name}")
    with open(
        f"{outputs_dir}/{agent_type}/{model_name}/{id_experiment}.json", "a+"
    ) as f:
        tmp_point = {"idx": idx, "response": response, "prompt": prompt}
        if price != 0:
            tmp_point["price"] = price
        # save tmp_point in the json file
        f.write(json.dumps(tmp_point))
        f.write("\n")
    return
