import numpy as np
import os
import pdb
import json
import unicodedata
import re
import copy


def normalize_name(name: str) -> str:
    # Normalize to NFKD form (compatibility decomposition)
    normalized = unicodedata.normalize("NFKD", name)
    # Encode to ASCII bytes, ignoring characters that can't be encoded (i.e., remove diacritics)
    ascii_bytes = normalized.encode("ASCII", "ignore")
    # Decode back to string
    ascii_str = ascii_bytes.decode("utf-8")
    # Optional: lowercase and remove extra characters
    ascii_str = re.sub(r"[^a-zA-Z]", "", ascii_str)  # keep only letters
    return ascii_str.lower()


def get_id_experiment(args, default):
    """
    Manages the ID by incrementing it and writing it to a file.
    If the file does not exist, it initializes the ID to 0.
    """
    if args.id_experiment == "-1" and not args.develop:
        new_id = get_id_new(args)
        reduced_args = reduce_args(args, default)
        write_id(args, new_id, reduced_args)
        return new_id
    return args.id_experiment


def reduce_args(args, default):
    reduced_args = {}
    for k, v in vars(args).items():
        if not k in ["neptune_args", "train_args"]:
            if vars(default)[k] != v:
                reduced_args[k] = v
    return reduced_args


def get_id_new(args):
    if not os.path.exists(f"{args.outputs_dir}/ids"):
        os.makedirs(f"{args.outputs_dir}/ids")
    if not os.path.exists(f"{args.outputs_dir}/ids/ids_config"):
        os.makedirs(f"{args.outputs_dir}/ids/ids_config")
    players = np.load(f"{args.outputs_dir}/ids/players.npy", allow_pickle=True)
    register = open(f"{args.outputs_dir}/ids/register.txt", "r+").readlines()
    return normalize_name(players[len(register)][0])


def write_id(args, id_experiment, reduced_args):
    register_file = open(f"{args.outputs_dir}/ids/register.txt", "a+")
    register_file.write(f"{id_experiment}\n")
    register_file.close()
    with open(
        f"{args.outputs_dir}/ids/ids_config/{id_experiment}.json", "a+"
    ) as config_file:
        args_tmp = vars(copy.deepcopy(args))
        args_tmp.pop("train_args", None)
        args_tmp.pop("neptune_args", None)
        tmp_config = {
            "id_experiment": id_experiment,
            "reduced_args": reduced_args,
            "args": args_tmp,
        }
        config_file.write(json.dumps(tmp_config))
        config_file.write("\n")
