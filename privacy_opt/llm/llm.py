from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import torch
import json
import pdb
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

ATTENTION_LINEARS = ["k_proj", "q_proj", "v_proj", "o_proj"]


class LLM:
    """
    A class to represent a local language model (LLM) loaded with huggingface.
    Depending
    """

    def __init__(self, model_name: str, agent_type: str, temperature: float):
        """
        Initialize an LLM instance.

        Args:
            model_name (str): The name of the model (e.g., "gpt-4", "mistral-7b").
            provider (str): The provider (e.g., "openai", "huggingface", "anthropic").
            api_key (str, optional): API key if required by the provider.
            params (dict, optional): Additional parameters like temperature, max_tokens, etc.
        """
        self.model_name = model_name
        self.agent_type = agent_type
        self.temperature = temperature
        # self.load_config()
        self.load_llm()

    # def load_config(self):
    #    with open("privacy_evals/local_models/config.json", "r") as f:
    #        config = json.load(f)[self.agent_type]
    #    self.prompt_template_path = config["prompt_template_path"]
    #    self.params = config["variables"]

    def load_llm(self):
        """
        Load the weights of the model.
        """

        bnb_quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=bnb_quantization_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=ATTENTION_LINEARS,  # For LLaMA; adjust for other models #["q_proj", "v_proj"]
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(model, lora_config)
        self.device = self.model.device
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def datapoint_to_inputs(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        if self.device == "cpu":
            return ""
        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
        )
        inputs["attention_mask"] = inputs["attention_mask"].to(self.device)
        inputs["input_ids"] = inputs["input_ids"].to(self.device)
        return inputs

    def generate_response(self, inputs):
        if self.device == "cpu":
            return "Placeholder for LLM response - No GPU available"
        try:
            # we don't sample
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=4000)
            tokenized_output = self.tokenizer.decode(out[0], skip_special_tokens=False)
        except:
            print("Error in model.generate")
            return
        return tokenized_output
