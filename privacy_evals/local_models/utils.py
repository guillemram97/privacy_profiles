from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import torch


def load_local_llm(args):
    bnb_quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    # Only use this if I am struggling!
    max_memory = {
        0: "40GiB",
        1: "0GiB",
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        quantization_config=bnb_quantization_config,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    return model, tokenizer


def prompt_to_chat_format(prompt):
    return [{"role": "user", "content": prompt}]


def tokenize(messages, tokenizer):
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
    )
    inputs["attention_mask"] = inputs["attention_mask"].to("cuda")
    inputs["input_ids"] = inputs["input_ids"].to("cuda")


def make_output(inputs, model, tokenizer):
    try:
        with torch.no_grad():
            out = model.generate(**inputs, temperature=0.000001)
        tokenized_output = tokenizer.decode(out[0], skip_special_tokens=True)
    except:
        print("Error in model.generate")
    return tokenized_output
