import secrets
import json
import os
import hashlib
import pdb

def deterministic_id(*args):
    s = '|'.join(str(arg) if arg is not None else 'None' for arg in args)
    hash_bytes = hashlib.sha256(s.encode()).digest()
    return hash_bytes[:3].hex()

def write_id_experiment(
    id_experiment: str,
    agent_type: str,
    outputs_dir: str,
    model_name: str,
    prompt_template: str,
    persona: str,
    prompt_variation: str,
    evaluated1: dict = None,
    evaluated2: dict = None,
) -> str:
    if id_experiment == "-1" and agent_type in ["answer", "aggregator", "deferral_post"]:
        return -1, [{}]
    elif id_experiment == "-1" and agent_type in ["paraphraser", "evaluator", "deferral"]:
        #id_experiment = secrets.token_hex(3)
        id_experiment = deterministic_id(agent_type, model_name, prompt_template, persona, prompt_variation, evaluated1, evaluated2)
    elif id_experiment == "-1":
        raise ValueError("Wrong id for the experiment!")
    if not os.path.exists(f"{outputs_dir}/ids_config"):
        os.makedirs(f"{outputs_dir}/ids_config")
    if agent_type == "evaluator":
        if not os.path.exists(f"{outputs_dir}/ids_config/evaluator"):
            os.makedirs(f"{outputs_dir}/ids_config/evaluator")
        config_path = f"{outputs_dir}/ids_config/evaluator/{id_experiment}.json"
    else:
        config_path = f"{outputs_dir}/ids_config/{id_experiment}.json"
    with open(config_path, "a+") as f:
        tmp_config = {
            "agent_type": agent_type,
            "model_name": model_name,
            "prompt_template": prompt_template,
            "prompt_variation": prompt_variation,
            "persona": persona,
        }
        if agent_type == "evaluator":
            tmp_config["evaluated1"] = evaluated1
            tmp_config["evaluated2"] = evaluated2
        f.write(json.dumps(tmp_config))
        f.write("\n")
    return id_experiment, read_id_experiment(id_experiment, outputs_dir, agent_type)


def read_id_experiment(id_experiment: str, outputs_dir: str, agent_type: str) -> None:
    # read the json. The file contains multiple lines.
    if agent_type == "evaluator":
        config_path = f"{outputs_dir}/ids_config/evaluator/{id_experiment}.json"
    else:
        config_path = f"{outputs_dir}/ids_config/{id_experiment}.json"
    with open(config_path, "r") as f:
        lines = f.readlines()
        config = []
        for line in lines:
            tmp_config = json.loads(line)
            config.append(tmp_config)
    return config


def correct_model_name(agent_type, experiment_id, model_name, outputs_dir):
    """
    If experiment_id is specified, we correct the model_name to match the one of the agent.
    If we evaluate an "answer" agent, then experiment_id should be -1.
    If we evaluate a non-"answer" agent, we should be able to retrieve the model name from the config.
    """
    if experiment_id != "-1":
        config = read_id_experiment(experiment_id, outputs_dir, agent_type)
        for x in config:
            if x["agent_type"] == agent_type:
                x["model_name"]
    return model_name


def update_cost_experiment(
    id_experiment: str,
    agent_type: str,
    model_name: str,
    prompt_template: str,
    outputs_dir: str,
    cost: float,
) -> str:
    cost_path = f"{outputs_dir}/ids_config/cost.json"
    with open(cost_path, "a+") as f:
        tmp_config = {
            "agent_type": agent_type,
            "id_experiment": id_experiment,
            "model_name": model_name,
            "prompt_template": prompt_template,
            "cost": cost,
        }
        if agent_type == "evaluator":
            tmp_config["evaluated1"] = evaluated1
            tmp_config["evaluated2"] = evaluated2
        f.write(json.dumps(tmp_config))
        f.write("\n")
    return