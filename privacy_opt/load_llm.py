from unsloth import (
    FastLanguageModel,
)
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast as Tokenizer
from peft.peft_model import PeftModelForCausalLM as Model
import pdb


# probably we want to pass the LoRA params as an argument!
def load_llm_unsloth(
    model_name: str, max_seq_length: int, lora_r: int, seed: int
) -> tuple[Model, Tokenizer]:
    if "uniner" in model_name.lower():
        model_name = "./UniNER-7B-all-safetensors"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
        device_map="balanced",
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=2 * lora_r,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        max_seq_length=max_seq_length,
        use_rslora=False,
        loftq_config=None,
    )
    return model, tokenizer
