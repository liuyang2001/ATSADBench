import os
from openai import OpenAI
from config import MODEL_CONFIG

class ModelHandler:
    def __init__(self, model_name):
        self.model_name = model_name
        self.config = MODEL_CONFIG.get(model_name, {})
        self.api_key = self.config.get("api_key")
        self.endpoint = self.config.get("endpoint")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.endpoint
        )

    def query(self, prompt):
        try:
            messages = [
                {"role": "system", "content": "You are an exceptionally intelligent assistant that detects anomalies in satellite telemetry time series data by listing all the anomalies. "},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=0
                # extra_body={"enable_thinking": False}#for Qwen3
            )
            print(f"prompt_tokens:{response.usage.prompt_tokens}")
            print(f"completion_tokens:{response.usage.completion_tokens}")
            print(f"total_tokens:{response.usage.total_tokens}")
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error querying {self.model_name}: {e}")
            return "Error querying"

def get_model_handler(model_name):
    return ModelHandler(model_name)