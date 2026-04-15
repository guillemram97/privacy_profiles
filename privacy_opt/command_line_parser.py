import argparse
from argparse import Namespace
import sys
import pdb

# import os
# os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
dic_models = {
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit": "meta-llama/Llama-3.1-8B-Instruct",
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit": "meta-llama/Llama-3.3-70B-Instruct",
    "unsloth/DeepSeek-R1-Distill-Llama-70B-unsloth-bnb-4bit": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "unsloth/DeepSeek-R1-Distill-Qwen-32B-unsloth-bnb-4bit": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit": "mistralai/Mistral-7B-Instruct-v0.3",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit": "meta-llama/Llama-3.2-3B-Instruct",
    "unsloth/gemma-3-27b-it-bnb-4bit": "google/gemma-3-27b-it",
    "unsloth/gemma-2-27b-it-bnb-4bit": "google/gemma-2-27b-it",
    "unsloth/Qwen3-32B-bnb-4bit": "qwen/Qwen-3-32B",
    "unsloth/Qwen3-30B-A3B-bnb-4bit": "qwen/Qwen-3-30B-A3B",
    "Universal-NER/UniNER-7B-all": "Universal-NER/UniNER-7B-all",
}

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        help="The name of the LLM.",
    )
    parser.add_argument(
        "--model_name_local",
        type=str,
        default="",
        help="The name of the LLM.",
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="aggregator",
        help="Type of agent: rejector, paraphraser, aggregator.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
    )
    parser.add_argument(
        "--develop",
        action="store_true",
        help="If true, it will run the code in develop mode, which doesnt register the run",
    )
    parser.add_argument(
        "--emoji_prompt",
        action="store_true",
        help="If true, it will use an emoji prompt.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="The rank of the LoRA adapter.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="The split of the dataset to use for testing. Options: train, val, test."
    )

    parser.add_argument(
        "--num_datapoints",
        type=int,
        default=100000,
        help="",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Either tatoeba or gsm8k.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num_epochs",
        type=float,
        default=10,
        help="Number of epochs for training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="checkpoints",
        help="Directory where checkpoints are stored.",
    )
    parser.add_argument(
        "--id_experiment",
        type=str,
        default="-1",
        help="ID of the experiment. If the model is answer, it should be -1 by default - unless it's the larger LLM.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for running the experiments.",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default="",
        help="Neptune tags. String delimited by ,",
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="outputs",
        help="Directory where outputs are stored.",
    )
    parser.add_argument(
        "--outputs_dir_checkpoints",
        type=str,
        default="outputs",
        help="Directory where checkpoints are stored.",
    )
    parser.add_argument(
        "--outputs_dir_pipeline", 
        type=str,
        default="/mnt_path/PrivacyEval/responses",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="Proportion of the dataset to include in the test + val split.",
    )
    parser.add_argument(
        "--id_experiment_prev",
        type=str,
        default="",
        help="ID of the previous experiment to load responses from.",
    )
    parser.add_argument(
        "--model_answer_post",
        type=str,
        default="gpt-4o",
    )
    parser.add_argument(
        "--inference_mode",
        type=str,
        default="pipeline",
        help="If equal to test, then it loads a custom dataset.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = create_parser()
    args = parser.parse_args(sys.argv[1:])
    default = parser.parse_args([])

    args.train_args = Namespace(
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        early_stopping_patience=args.early_stopping_patience,
        outputs_dir_checkpoints=args.outputs_dir_checkpoints,
        tags=args.tags,
    )
    args.neptune_args = Namespace(
        lora_r=args.lora_r,
        agent_type=args.agent_type,
        model_name=args.model_name,
    )
    args.model_name_local = dic_models[args.model_name]
    return args, default
