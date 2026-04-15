from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import torch
import json
import pdb


class LocalModel:
    """
    A class to represent a local language model (LLM) loaded with huggingface.
    Depending
    """

    def __init__(self, model_name: str, agent_type: str, temperature: float, device: int, prompt_variation: str):
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
        self.device = device
        self.prompt_variation = prompt_variation
        self.load_config()
        self.load_weights()
        self.load_template_prompt()

    def load_config(self):
        with open("privacy_evals/local_models/config.json", "r") as f:
            config = json.load(f)[self.agent_type]
        self.prompt_template_path = config["prompt_template_path"]
        if self.prompt_variation != "":
            self.prompt_template_path = config["prompt_template_path"].replace(
                ".txt", f"_{self.prompt_variation}.txt"
            )
        self.params = config["variables"]

    def load_weights(self):
        """
        Load the weights of the model.
        """
        if not torch.cuda.is_available():
            self.device = "cpu"
            self.tokenizer = ""
            self.model = ""
            return
        bnb_quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        if 'openai' in self.model_name:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=f"cuda:{self.device}",
            )
        elif self.model_name in ['mistralai/Mistral-Small-3.2-24B-Instruct-2506', 'mistralai/Mistral-Small-3.1-24B-Instruct-2503']:
            from transformers import MistralForCausalLM, LlamaTokenizerFast
            self.model = MistralForCausalLM.from_pretrained(
                self.model_name,
                device_map=f"cuda:{self.device}",
                attn_implementation="flash_attention_2",
                quantization_config=bnb_quantization_config,
            )
            self.tokenizer = LlamaTokenizerFast.from_pretrained('mistralai/Mistral-Small-Instruct-2409')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            if self.device == -1:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map=f"cpu",
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map=f"cuda:{self.device}",
                    quantization_config=bnb_quantization_config,
                    attn_implementation="flash_attention_2",
                )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = self.model.device

    def load_template_prompt(self):
        """
        Load the template prompt for the model.
        """
        self.template_prompt = "".join(open(self.prompt_template_path, "r").readlines())

    def make_prompts(self, data):
        for idx in range(len(data)):
            aux = data.iloc[idx]
            var = {param: aux[param] for param in self.params}
            data.at[idx, "prompt"] = self.template_prompt.format(**var)
        return data

    def make_prompt_evaluation(self, data, inv=False):
        if inv:
            return self.template_prompt.format(
                query=data["query"],
                response_A=data["response_B"],
                response_B=data["response_A"],
            )
        return self.template_prompt.format(
            query=data["query"],
            response_A=data["response_A"],
            response_B=data["response_B"],
        )

    def datapoint_to_inputs(self, prompt):
        prompt = str(prompt)
        if 'gpt' in self.model_name:
            messages = [{"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2025-06-28\nReasoning: low\n# Valid channels: analysis, commentary, final. Channel must be included for every message.\nCalls to these tools must go to the commentary channel"}, {"role": "user", "content": prompt}]
        else:
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
            attempt = 0
            stop = False
            while attempt < 5 and not stop:
                attempt += 1
                with torch.no_grad():
                    out = self.model.generate(**inputs, max_new_tokens=4000, temperature=self.temperature, do_sample=True)
                tokenized_output = self.tokenizer.decode(out[0], skip_special_tokens=False)
                if self.agent_type == "paraphraser":
                    if "[[[ ### createdPrompt ### ]]]" in tokenized_output or 'gpt' in self.model_name:
                        stop = True
                elif self.agent_type == "deferral" or self.agent_type == "deferral_post": 
                    if "[[[ ### label ### ]]]" in tokenized_output:
                        stop = True
                else: stop = True
            if not stop:
                tokenized_output = "[[Not found.]]"
        except:
            print("Error in model.generate")
            return
        return tokenized_output
