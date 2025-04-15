from lib.model import TextGenerator
from lib.generation_utils import apply_temperature, get_top_p_tokens, normalize_logits
from lib.logger import setup_logger
import torch


def nucleus_sample_decode(generator, prompt, temperature, top_p):
    inputs = generator.prepare_inputs(prompt)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    generated_ids = []

    for _ in range(generator.max_length):
        logits = generator.get_next_token_logits(input_ids, attention_mask)
        logits = apply_temperature(logits, temperature)
        logits = get_top_p_tokens(logits, top_p)
        probs = normalize_logits(logits)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated_ids.append(next_token)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
        attention_mask = torch.cat([attention_mask, torch.tensor([[1]])], dim=1)
        if next_token == generator.eos_token_id:
            break

    return generated_ids


def main():
    logger = setup_logger('task4')
    generator = TextGenerator()
    parameters = [
        (1.0, 0.9),
        (1.0, 0.15),
        (0.5, 0.9),
        (0.5, 0.15)
    ]
    logger.info("Generating hedgehog stories with different parameters:")
    for temp, top_p in parameters:
        hedgehog_prompt = generator.get_hedgehog_prompt()
        hedgehog_ids = nucleus_sample_decode(generator, hedgehog_prompt, temp, top_p)
        hedgehog_text = generator.decode_output(hedgehog_ids)
        logger.info(f"\nTemperature: {temp}, Top-p: {top_p}")
        logger.info(hedgehog_text)
        logger.info("\n" + "=" * 50 + "\n")
    logger.info("Generating JSON outputs with different parameters:")
    for temp, top_p in parameters:
        json_prompt = generator.get_json_prompt()
        json_ids = nucleus_sample_decode(generator, json_prompt, temp, top_p)
        json_text = generator.decode_output(json_ids)
        logger.info(f"\nTemperature: {temp}, Top-p: {top_p}")
        logger.info(json_text)
        logger.info("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
