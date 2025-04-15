from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class TextGenerator:
    def __init__(self, model_name='Qwen/Qwen2.5-0.5B-Instruct'):
        self.model = AutoModelForCausalLM.from_pretrained(model_name).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.eos_token_id = 151645
        self.max_length = 1000

    def get_hedgehog_prompt(self):
        return '<|im_start|>system\nYou are a storyteller. Generate a story based on user message.<|im_end|>\n<|im_start|>user\nGenerate me a short story about a tiny hedgehog named Sonic.<|im_end|>\n<|im_start|>assistant\n'

    def get_json_prompt(self):
        return '<|im_start|>system\nYou are a JSON machine. Generate a JSON with format {"contractor": string with normalized contractor name, "sum": decimal, "currency": string with uppercased 3-letter currency code} based on user message.<|im_end|>\n<|im_start|>user\nTransfer 100 rubles and 50 kopeck to Mike<|im_end|>\n<|im_start|>assistant\n'

    def prepare_inputs(self, prompt):
        encoding = self.tokenizer(prompt).encodings[0]
        return {
            'input_ids': torch.tensor([encoding.ids]),
            'attention_mask': torch.tensor([encoding.attention_mask])
        }

    def decode_output(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def get_next_token_logits(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.logits[0, -1]
