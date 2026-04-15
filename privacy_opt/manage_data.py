from unsloth import (
    to_sharegpt,
    standardize_sharegpt,
    apply_chat_template,
    get_chat_template,
)
from datasets import load_dataset, DatasetDict
import pdb
import json
import os
import pandas as pd
from transformers.tokenization_utils_fast import (
    PreTrainedTokenizerFast as FastTokenizer,
)
import transformers

CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{OUTPUT}<|eot_id|>"""

SYSTEM_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM}"""

def load_data(
    agent_name: str,
    num_datapoints: int,
    tokenizer: FastTokenizer,
    test_split: float,
    seed: int,
) -> DatasetDict:

    dataset = load_dataset("csv", data_files=f"data/{agent_name}.csv")["train"]
    for feature in dataset.features:
        if not feature in ["x", "y"]:
            dataset = dataset.remove_columns([feature])

    dataset = dataset.select(range(min(num_datapoints, len(dataset))))

    to_sharegpt(
        dataset,
        merged_prompt="{x}",
        output_column_name="y",
        conversation_extension=1,
    )
    dataset = to_sharegpt(
        dataset,
        merged_column_name="x",
        output_column_name="y",
        conversation_extension=1,
    )

    dataset = standardize_sharegpt(dataset)

    try:
        if 'mistral' in tokenizer.vocab_file:

            def formatting_prompts_func(examples):
                convos = examples["conversations"]
                texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False)[3:] for convo in convos]
                return {"text" : texts,}
            
            tokenizer = get_chat_template(
                tokenizer,
                chat_template = "mistral",
            )
            dataset = dataset.map(formatting_prompts_func, batched = True,)
    except:
        dataset = apply_chat_template(
            dataset,
            tokenizer=tokenizer,
            chat_template=CHAT_TEMPLATE,
            # default_system_message,
        )
    if test_split == -1:
        train_test_valid_dataset = DatasetDict(
            {
                "train": dataset,
                "valid": dataset,
                "test": dataset,
            }
        )
    else:
        train_testvalid = dataset.train_test_split(test_size=test_split, seed=seed)
        test_valid = train_testvalid["test"].train_test_split(test_size=0.5, seed=seed)
        train_test_valid_dataset = DatasetDict(
            {
                "train": train_testvalid["train"],
                "valid": test_valid["train"],
                "test": test_valid["test"],
            }
        )

    return train_test_valid_dataset



def load_data_no_conv(
    agent_name: str,
    num_datapoints: int,
    tokenizer: FastTokenizer,
    test_split: float,
    seed: int,
) -> DatasetDict:

    dataset = load_dataset("csv", data_files=f"data/{agent_name}.csv")["train"]
    for feature in dataset.features:
        if not feature in ["x", "y"]:
            dataset = dataset.remove_columns([feature])
    
    dataset = dataset.select(range(min(num_datapoints, len(dataset))))

    

    MAX_LEN = tokenizer.model_max_length  # should be 4096 for LLaMA

    def preprocess(example):
        instruction = example["x"]
        output = example["y"]

        full_text = instruction + output

        input_ids = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LEN,
        ).input_ids

        instruction_ids = tokenizer(
            instruction,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LEN,
        ).input_ids

        labels = [-100] * len(instruction_ids) + input_ids[len(instruction_ids):]

        labels = labels[:len(input_ids)]

        assert len(input_ids) == len(labels)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }
    
    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)


    if test_split == -1:
        train_test_valid_dataset = DatasetDict(
            {
                "train": dataset,
                "valid": dataset,
                "test": dataset,
            }
        )
    else:
        train_testvalid = dataset.train_test_split(test_size=test_split, seed=seed)
        test_valid = train_testvalid["test"].train_test_split(test_size=0.5, seed=seed)
        train_test_valid_dataset = DatasetDict(
            {
                "train": train_testvalid["train"],
                "valid": test_valid["train"],
                "test": test_valid["test"],
            }
        )

    return train_test_valid_dataset





def prepare_inference_data(test):
    for idx, item in enumerate(test):
        if type(item) == list:
            test[idx] = item[0]['content']
        else:
            test[idx] = item[:item.find('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')+len('<|eot_id|><|start_header_id|>assistant<|end_header_id|>')]
    return test

def load_data_pipeline(
    num_datapoints: int,
    agent_type: str,
    id_experiment: str,
    outputs_dir: str,
    id_experiment_prev: str,
    model_answer_post: str = None,
    model_name_local: str = None,
    config: list[dict] = None,
) -> pd.DataFrame:
    #split = "real"

    if len(agent_type.split("_")) > 1 and agent_type.split("_")[1] == "untrained":
        split = "train"
    #if len(agent_type.split("_")) > 1 and "real" in agent_type.split("_")[1]:
    #    split = "real"
    split = "test"
    data = pd.read_csv(f"data/peep_{split}.csv", nrows=num_datapoints)
    ## AQUI ES ON FAIG LO BO!
    if agent_type == "answer" and id_experiment == "-1":
        return data
    if id_experiment == "-1":
        raise ValueError("Wrong id for the experiment!")
    if agent_type == "paraphraser" or agent_type == "deferral" or agent_type == "paraphraser_untrained":
        return data
    if agent_type == "answer_post" or agent_type == "deferral_post":
        agent_type_tmp = "paraphraser"
        model_name_tmp = model_name_local
        id_tmp, response_tmp, _ = load_response_pipeline(
            outputs_dir, model_name_tmp, agent_type_tmp, id_experiment_prev
        )
        to_drop = []
        for idx in range(len(data)):
            aux = data.iloc[idx]
            if int(aux['idx_wildchat']) in id_tmp:
                data.at[idx, "query_modified"] = response_tmp[list(id_tmp).index(int(aux['idx_wildchat']))]
            else:
                to_drop.append(idx)
        data = data.drop(to_drop).reset_index(drop=True)  
    if agent_type == "aggregator":
        agent_type_tmp = "answer_post"
        model_name_tmp = model_answer_post
        id_tmp, response_tmp, prompts_tmp = load_response_pipeline(
            outputs_dir, model_name_tmp, agent_type_tmp, id_experiment_prev
        )
        to_drop = []
        for idx in range(len(data)):
            aux = data.iloc[idx]
            if int(aux['idx_wildchat']) in id_tmp or str(aux['idx_wildchat']) in id_tmp:
                data.at[idx, "response"] = response_tmp[list(id_tmp).index(int(aux['idx_wildchat']))]
                data.at[idx, "query_modified"] = prompts_tmp[list(id_tmp).index(int(aux['idx_wildchat']))]
            else:
                to_drop.append(idx)
        data = data.drop(to_drop).reset_index(drop=True)  
    return data




def load_data_pipeline_extra_dataset(
    num_datapoints: int,
    agent_type: str,
    id_experiment: str,
    outputs_dir: str,
    id_experiment_prev: str,
    model_answer_post: str = None,
    model_name_local: str = None,
    config: list[dict] = None,
    dataset_name: str = "gsm8k",
) -> pd.DataFrame:
    #split = "real"

    if len(agent_type.split("_")) > 1 and agent_type.split("_")[1] == "untrained":
        split = "train"
    split = "test"
    if dataset_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main")
        ds = ds[split].to_pandas()
        data = ds.rename(columns={"question": "query", "answer": "gold_response"})
        data['idx_wildchat'] = data.index
        data['low_relevance'] = False
        data['profile'] = "Don't share information about names, locations, occupations, email addresses or any other personally identifiable information."
        data['language'] = 'English'
        prompt_template = "Question: {question} \n Answer with step-by-step reasoning, and write the final number after ####"
        for idx in range(len(data)):
            aux = data.iloc[idx]['query']
            data.at[idx, 'query'] = prompt_template.format(question = aux)

    elif dataset_name == "tatoeba":
        data = load_dataset("tatoeba", lang1="en", lang2="eu", trust_remote_code=True)['train'].to_pandas()
        data['idx_wildchat'] = data.index
        prompt_template = "Translate the English sentence into Basque. English sentence: {src} \n Basque sentence:"
        for idx in range(len(data)):
            aux = data.iloc[idx]['translation']
            data.at[idx, 'query'] = prompt_template.format(src = aux['en'])
            data.at[idx, 'gold_response'] = aux['eu']
        data['low_relevance'] = False
        data['profile'] = "Don't share information about names, locations, occupations, email addresses or any other personally identifiable information."
        data['language'] = 'English'
    if agent_type == "answer" and id_experiment == "-1":
        return data
    if id_experiment == "-1":
        raise ValueError("Wrong id for the experiment!")
    if agent_type == "paraphraser" or agent_type == "deferral" or agent_type == "paraphraser_untrained":
        return data
    if agent_type == "answer_post" or agent_type == "deferral_post":
        agent_type_tmp = "paraphraser"
        model_name_tmp = model_name_local
        id_tmp, response_tmp, _ = load_response_pipeline_alt_dataset(
            outputs_dir, model_name_tmp, agent_type_tmp, id_experiment_prev, dataset_name
        )
        to_drop = []
        for idx in range(len(data)):
            aux = data.iloc[idx]
            if int(aux['idx_wildchat']) in id_tmp:
                data.at[idx, "query_modified"] = response_tmp[list(id_tmp).index(int(aux['idx_wildchat']))]
            else:
                to_drop.append(idx)
        data = data.drop(to_drop).reset_index(drop=True)  
    if agent_type == "aggregator":
        agent_type_tmp = "answer_post"
        model_name_tmp = model_answer_post
        id_tmp, response_tmp, prompts_tmp = load_response_pipeline_alt_dataset(
            outputs_dir, model_name_tmp, agent_type_tmp, id_experiment_prev, dataset_name,
        )
        to_drop = []
        for idx in range(len(data)):
            aux = data.iloc[idx]
            if int(aux['idx_wildchat']) in id_tmp or str(aux['idx_wildchat']) in id_tmp:
                data.at[idx, "response"] = response_tmp[list(id_tmp).index(int(aux['idx_wildchat']))]
                data.at[idx, "query_modified"] = prompts_tmp[list(id_tmp).index(int(aux['idx_wildchat']))]
            else:
                to_drop.append(idx)
        data = data.drop(to_drop).reset_index(drop=True)  
    return data



def load_response_pipeline(
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
                    #if "[[[ ### createdPrompt ### ]]]" in tmp_point["response"]:
                    ids_tmp.append(tmp_point["idx"])
                    response_tmp.append(tmp_point["response"])
                elif agent_type == "deferral":
                    if "[[[ ### label ### ]]]" in tmp_point["response"]:
                        ids_tmp.append(tmp_point["idx"])
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

def load_response_pipeline_alt_dataset(
    outputs_dir: str, model_name: str, agent_type: str, id_experiment: str, dataset_name: str
):
    try:
        f = open(f"{outputs_dir}/{dataset_name}/{agent_type}/{model_name}/{id_experiment}.json", "r")
        lines = f.readlines()
        ids_tmp = []
        response_tmp = []
        prompts_tmp = []
        
        for line in lines:
            try:
                tmp_point = json.loads(line)
                if agent_type == "paraphraser":
                    #if "[[[ ### createdPrompt ### ]]]" in tmp_point["response"]:
                    ids_tmp.append(tmp_point["idx"])
                    response_tmp.append(tmp_point["response"])
                elif agent_type == "deferral":
                    if "[[[ ### label ### ]]]" in tmp_point["response"]:
                        ids_tmp.append(tmp_point["idx"])
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
    


def save_output_pipeline(
    outputs_dir: str,
    idx: int,
    response: str,
    prompt: str,
    model_name: str,
    agent_type: str,
    id_experiment: str,
    price: float = 0,
) -> None:
    #agent_type = agent_type.split("_")[0]
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
