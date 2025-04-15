from lib.model import TextGenerator
from lib.logger import setup_logger
import torch


def greedy_decode(generator, prompt):
    inputs = generator.prepare_inputs(prompt)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    generated_ids = []

    for _ in range(generator.max_length):
        logits = generator.get_next_token_logits(input_ids, attention_mask)
        next_token = torch.argmax(logits, dim=-1).item()
        generated_ids.append(next_token)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
        attention_mask = torch.cat([attention_mask, torch.tensor([[1]])], dim=1)
        if next_token == generator.eos_token_id:
            break

    return generated_ids


def main():
    logger = setup_logger('task1')
    generator = TextGenerator()
    hedgehog_prompt = generator.get_hedgehog_prompt()
    hedgehog_ids = greedy_decode(generator, hedgehog_prompt)
    hedgehog_text = generator.decode_output(hedgehog_ids)
    logger.info("Hedgehog Story:")
    logger.info(hedgehog_text)
    logger.info("\n" + "=" * 50 + "\n")
    json_prompt = generator.get_json_prompt()
    json_ids = greedy_decode(generator, json_prompt)
    json_text = generator.decode_output(json_ids)
    logger.info("JSON Output:")
    logger.info(json_text)


if __name__ == "__main__":
    main()
