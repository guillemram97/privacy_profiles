from openai import OpenAI
import openai
import torch
import json
import pdb
import backoff
import conf.secrets as sc
OPENAI_API_KEY = sc.OPENAI_API_KEY


class ExternalModel:
    """
    A class to represent a local language model (LLM) loaded with huggingface.
    Depending
    """

    def __init__(self, model_name: str, agent_type: str, temperature: float, prompt_variation: str):
        """
        Initialize an LLM instance.

        Args:
            model_name (str): The name of the model (e.g., "gpt-4", "mistral-7b").
        """
        self.model_name = model_name
        self.agent_type = agent_type
        self.temperature = temperature
        self.prompt_variation = prompt_variation
        self.load_config()
        self.load_template_prompt()
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.cost = self.get_cost_scheme()

    def load_config(self):
        with open("privacy_evals/local_models/config.json", "r") as f:
            config = json.load(f)[self.agent_type]
        self.prompt_template_path = config["prompt_template_path"]
        if self.prompt_variation != "":
            self.prompt_template_path = config["prompt_template_path"].replace(
                ".txt", f"_{self.prompt_variation}.txt"
            )
        self.params = config["variables"]

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
        messages = [{"role": "user", "content": prompt}]
        return messages
    
    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def chat_completions_with_backoff(self, model_name, messages, temperature):
        return self.client.chat.completions.create(model=model_name, messages=messages, temperature=temperature)

    def generate_response(self, inputs):
        completion = self.chat_completions_with_backoff(model_name=self.model_name, messages=inputs, temperature=self.temperature)
        prompt_tokens = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens
        price = (
            prompt_tokens * self.cost["prompt"] + completion_tokens * self.cost["completion"]
        )/1e6
        response = completion.choices[0].message.content
        return response, price

    def get_cost_scheme(self):
        """
        Get the cost scheme for the model.
        """
        return {
            "gpt-4": {"prompt": 2.5, "completion": 10},
            "gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},
        }[self.model_name]
