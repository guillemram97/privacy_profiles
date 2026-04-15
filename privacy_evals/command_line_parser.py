import argparse
from .utils import utils


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="The name of the local LLM.",
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="answer",
        help="Type of agent: answer, paraphraser, aggregator.",
    )
    parser.add_argument(
        "--model_answer_post",
        type=str,
        default="gpt-4o-mini",
        help="If the agent is an aggregator, the model used to generate the answer.",
    )
    parser.add_argument(
        "--persona",
        type=str,
        default="basic",
        help="Persona of the privacy profiles.",
    )
    parser.add_argument(
        "--prompt_variation",
        type=str,
        default="",
        help="To test different types of prompts.",
    )
    parser.add_argument(
        "--num_datapoints",
        type=int,
        default=20000,
        help="",
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        default=1,
        help="Into how many splits the dataset should be divided.",
    )
    parser.add_argument(
        "--n_init",
        type=int,
        default=0,
        help="Number of the split (orderred).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature of the local LLM.",
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="responses",
        help="Directory where probabilities vectors are stored.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Number of the GPU to use.",
    )
    parser.add_argument(
        "--id_experiment",
        type=str,
        default="-1",
        help="ID of the experiment. If the model is answer, it should be -1 by default - unless it's the larger LLM.",
    )
    parser.add_argument(
        "--evaluator_target_1",
        type=str,
        default="answer",
        help="First model to be evaluated.",
    )
    parser.add_argument(
        "--evaluator_target_1_id",
        type=str,
        default="-1",
        help="Id of the first model to be evaluated.",
    )
    parser.add_argument(
        "--evaluator_target_1_model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Name of the first model to be evaluated.",
    )
    parser.add_argument(
        "--evaluator_target_2",
        type=str,
        default="answer",
        help="Second model to be evaluated.",
    )
    parser.add_argument(
        "--evaluator_target_2_id",
        type=str,
        default="-1",
        help="Id of the second model to be evaluated.",
    )
    parser.add_argument(
        "--evaluator_target_2_model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Name of the second model to be evaluated.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = create_parser()
    args = parser.parse_args()

    args.evaluator_target_1_model_name = utils.correct_model_name(
        args.evaluator_target_1,
        args.evaluator_target_1_id,
        args.evaluator_target_1_model_name,
        args.outputs_dir,
    )
    args.evaluator_target_2_model_name = utils.correct_model_name(
        args.evaluator_target_2,
        args.evaluator_target_2_id,
        args.evaluator_target_2_model_name,
        args.outputs_dir,
    )
    return args
