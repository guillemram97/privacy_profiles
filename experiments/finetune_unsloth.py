from unsloth import unsloth_train
import unsloth
from privacy_opt import command_line_parser, load_llm, manage_data, manage_trainer
from privacy_opt.utils import ids
import pdb

"""
    Useful resources:
    https://github.com/unslothai/unsloth/discussions/2828
    https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing#scrollTo=jZxeGSeX0CR8
"""


def main(args, default):
    args.id_experiment = ids.get_id_experiment(args, default)
    model, tokenizer = load_llm.load_llm_unsloth(
        args.model_name, args.max_seq_length, args.lora_r, args.seed
    )
    #dataset = manage_data.load_data(
    #    args.agent_type, args.num_datapoints, tokenizer, args.test_split, args.seed
    #)
    if 'UniNER' in args.model_name:
        dataset = manage_data.load_data_no_conv(
            args.agent_type, args.num_datapoints, tokenizer, args.test_split, args.seed
        )
    else:
        dataset = manage_data.load_data(
            args.agent_type, args.num_datapoints, tokenizer, args.test_split, args.seed
        )
    trainer = manage_trainer.get_trainer(args.train_args, model, tokenizer, dataset)
    manage_trainer.update_neptune(trainer, args)

    trainer_stats = unsloth_train(trainer)

    if "uniner" in args.model_name.lower():
        # now we need to deal with a funny merge
        from unsloth import FastLanguageModel
        import torch
        from peft import PeftModel
        model_a, tokenizer = FastLanguageModel.from_pretrained( "./UniNER-7B-all-safetensors", dtype=torch.float16,)
        model = PeftModel.from_pretrained(model_a, trainer.state.best_model_checkpoint)
        model = model.merge_and_unload()
        model.save_pretrained(f"best_models/{args.id_experiment}", safe_serialization=True)
        tokenizer.save_pretrained(f"best_models/{args.id_experiment}")
        #model.save_pretrained_merged(
        #    f"best_models/{args.id_experiment}_alt",
        #    tokenizer,
        #    save_method="merged_16bit",
        #)
    else:
        model.save_pretrained_merged(
            f"best_models/{args.id_experiment}",
            tokenizer,
            save_method="merged_16bit",
        )

if __name__ == "__main__":
    args, default = command_line_parser.parse_args()
    main(args, default)