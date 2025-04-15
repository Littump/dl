from lib.model import TextGenerator
from lib.generation_utils import apply_temperature, normalize_logits
from lib.logger import setup_logger
import torch


def temperature_sample_decode(generator, prompt, temperature):
    inputs = generator.prepare_inputs(prompt)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    generated_ids = []

    for _ in range(generator.max_length):
        logits = generator.get_next_token_logits(input_ids, attention_mask)
        logits = apply_temperature(logits, temperature)
        probs = normalize_logits(logits)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated_ids.append(next_token)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
        attention_mask = torch.cat([attention_mask, torch.tensor([[1]])], dim=1)
        if next_token == generator.eos_token_id:
            break

    return generated_ids


def main():
    logger = setup_logger('task3')
    generator = TextGenerator()
    temperatures = [0.001, 0.1, 0.5, 1.0, 10.0]

    logger.info("Generating hedgehog stories with different temperatures:")
    for temp in temperatures:
        hedgehog_prompt = generator.get_hedgehog_prompt()
        hedgehog_ids = temperature_sample_decode(generator, hedgehog_prompt, temp)
        hedgehog_text = generator.decode_output(hedgehog_ids)
        logger.info(f"\nTemperature: {temp}")
        logger.info(hedgehog_text)
        logger.info("\n" + "="*50 + "\n")

    logger.info("Generating JSON outputs with different temperatures:")
    for temp in temperatures:
        json_prompt = generator.get_json_prompt()
        json_ids = temperature_sample_decode(generator, json_prompt, temp)
        json_text = generator.decode_output(json_ids)
        logger.info(f"\nTemperature: {temp}")
        logger.info(json_text)
        logger.info("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
