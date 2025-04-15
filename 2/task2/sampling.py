from lib.model import TextGenerator
from lib.generation_utils import normalize_logits
from lib.logger import setup_logger
import torch


def sample_decode(generator, prompt, temperature=1.0):
    inputs = generator.prepare_inputs(prompt)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    generated_ids = []

    for _ in range(generator.max_length):
        logits = generator.get_next_token_logits(input_ids, attention_mask)
        # Apply temperature scaling
        scaled_logits = logits / temperature
        probs = normalize_logits(scaled_logits)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated_ids.append(next_token)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
        attention_mask = torch.cat([attention_mask, torch.tensor([[1]])], dim=1)
        if next_token == generator.eos_token_id:
            break

    return generated_ids


def main():
    logger = setup_logger('task2')
    generator = TextGenerator()
    logger.info("Generating 3 hedgehog stories:")
    for i in range(3):
        hedgehog_prompt = generator.get_hedgehog_prompt()
        hedgehog_ids = sample_decode(generator, hedgehog_prompt)
        hedgehog_text = generator.decode_output(hedgehog_ids)
        logger.info(f"\nStory {i + 1}:")
        logger.info(hedgehog_text)
        logger.info("\n" + "=" * 50 + "\n")
    logger.info("Generating 3 JSON outputs:")
    for i in range(3):
        json_prompt = generator.get_json_prompt()
        json_ids = sample_decode(generator, json_prompt)
        json_text = generator.decode_output(json_ids)
        logger.info(f"\nJSON {i + 1}:")
        logger.info(json_text)
        logger.info("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
