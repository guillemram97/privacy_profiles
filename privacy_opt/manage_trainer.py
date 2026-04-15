from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig
import torch
from transformers import DataCollatorForLanguageModeling
import pdb
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast as Tokenizer
from peft.peft_model import PeftModelForCausalLM as Model
from argparse import Namespace
from datasets import DatasetDict


def build_compute_metrics(tokenizer):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    def compute_metrics(eval_preds, compute_result=None):
        # eval_preds is a tuple of (predictions, labels)
        # predictions are logits, labels are the ground truth
        # the logits are obtained through teacher forcing
        # it doesnt make a lot of sense to use logits here!
        predictions, labels = eval_preds
        predictions = predictions[0]
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(
            torch.where(labels == -100, torch.tensor(tokenizer.pad_token_id), labels),
            skip_special_tokens=True,
        )

        # Example BLEU score computation:
        smoothie = SmoothingFunction().method4
        bleu_scores = [
            sentence_bleu([label.split()], pred.split(), smoothing_function=smoothie)
            for pred, label in zip(decoded_preds, decoded_labels)
        ]
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        return {
            "bleu": avg_bleu,
        }

    return compute_metrics


def preprocess_logits_for_metrics(logits, labels):
    # This is a workaround to avoid storing too many tensors that are not needed.
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels


def get_trainer(
    args: Namespace, model: Model, tokenizer: Tokenizer, dataset: DatasetDict
) -> SFTTrainer:

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=0,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        # compute_metrics=compute_metrics,
        args=SFTConfig(
            per_device_train_batch_size=args.train_batch_size,  # I think 16 is too large!
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            eval_accumulation_steps=2,
            batch_eval_metrics=True,
            max_steps=300,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=10,
            # num_train_epochs=args.num_epochs,
            logging_strategy="steps",
            logging_steps=10,
            max_seq_length=args.max_seq_length,
            output_dir=args.outputs_dir_checkpoints,
            optim="adamw_8bit",
            seed=args.seed,
            save_strategy="steps",
            eval_steps=10,
            eval_strategy="steps",
            save_steps=10,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="loss",  # or your eval metric
            greater_is_better=False,
        ),
    )

    try:
        if 'mistral' in tokenizer.vocab_file:
            trainer = train_on_responses_only(trainer, instruction_part ='[INST]', response_part = '[/INST]')
    except:
        trainer = train_on_responses_only(trainer)
    trainer.add_callback(early_stopping_callback)

    return trainer

def get_trainer_no_chat(
    args: Namespace, model: Model, tokenizer: Tokenizer, dataset: DatasetDict
) -> SFTTrainer:

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=0,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        tokenizer=tokenizer,
        data_collator=None,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        # compute_metrics=compute_metrics,
        args=SFTConfig(
            per_device_train_batch_size=args.train_batch_size,  # I think 16 is too large!
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            eval_accumulation_steps=2,
            batch_eval_metrics=True,
            max_steps=300,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=6,
            # num_train_epochs=args.num_epochs,
            logging_strategy="steps",
            logging_steps=5,
            max_seq_length=args.max_seq_length,
            output_dir=args.outputs_dir_checkpoints,
            optim="adamw_8bit",
            seed=args.seed,
            save_strategy="steps",
            eval_steps=3,
            eval_strategy="steps",
            save_steps=3,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="loss",  # or your eval metric
            greater_is_better=False,
        ),
    )

    trainer.add_callback(early_stopping_callback)

    return trainer


def update_neptune(trainer: SFTTrainer, args: Namespace) -> None:
    for k, v in vars(args.neptune_args).items():
        trainer.callback_handler.callbacks[1].run[k] = v
    trainer.callback_handler.callbacks[1].run["id_experiment"] = args.id_experiment
    trainer.callback_handler.callbacks[1].run["sys/tags"].add(args.tags.split(","))
    return
