from lib.model import TextGenerator
from lib.logger import setup_logger
import torch
from dataclasses import dataclass
from typing import List


@dataclass
class Candidate:
    token_ids: List[int]
    score: float
    is_finished: bool = False


def beam_search_decode(generator, prompt, num_beams, length_penalty):
    inputs = generator.prepare_inputs(prompt)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    current_candidates = []
    finished_candidates = []

    logits = generator.get_next_token_logits(input_ids, attention_mask)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    top_k_log_probs, top_k_indices = torch.topk(log_probs, num_beams)

    for i in range(num_beams):
        token_id = top_k_indices[0][i].item()
        score = top_k_log_probs[0][i].item()
        is_finished = (token_id == generator.eos_token_id)

        candidate = Candidate(
            token_ids=[token_id],
            score=score,
            is_finished=is_finished
        )

        if is_finished:
            finished_candidates.append(candidate)
        else:
            current_candidates.append(candidate)

    while len(finished_candidates) < num_beams and current_candidates:
        new_candidates = []

        for candidate in current_candidates:
            candidate_input_ids = torch.cat([
                input_ids,
                torch.tensor([candidate.token_ids])
            ], dim=1)
            candidate_attention_mask = torch.cat([
                attention_mask,
                torch.ones(1, len(candidate.token_ids))
            ], dim=1)

            logits = generator.get_next_token_logits(
                candidate_input_ids,
                candidate_attention_mask
            )
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            top_k_log_probs, top_k_indices = torch.topk(log_probs, num_beams)

            for i in range(num_beams):
                token_id = top_k_indices[0][i].item()
                new_score = candidate.score + top_k_log_probs[0][i].item()
                is_finished = (token_id == generator.eos_token_id)

                new_candidate = Candidate(
                    token_ids=candidate.token_ids + [token_id],
                    score=new_score,
                    is_finished=is_finished
                )

                if is_finished:
                    finished_candidates.append(new_candidate)
                else:
                    new_candidates.append(new_candidate)

        current_candidates = sorted(
            new_candidates,
            key=lambda x: x.score / (len(x.token_ids) ** length_penalty),
            reverse=True
        )[:num_beams]

    all_candidates = finished_candidates + current_candidates

    # Если есть законченные кандидаты - выбираем только из них
    if finished_candidates:
        best_candidates = sorted(
            finished_candidates,
            key=lambda x: x.score / (len(x.token_ids) ** length_penalty),
            reverse=True
        )
    else:
        # Иначе берем из всех доступных
        best_candidates = sorted(
            all_candidates,
            key=lambda x: x.score / (len(x.token_ids) ** length_penalty),
            reverse=True
        )

    return best_candidates[0].token_ids


def main():
    logger = setup_logger('task5')
    generator = TextGenerator()
    parameters = [
        (1, 1.0),
        (4, 1.0),
        (4, 0.5),
        (4, 2.0),
        (8, 1.0)
    ]

    logger.info("Generating hedgehog stories with different parameters:")
    for num_beams, length_penalty in parameters:
        hedgehog_prompt = generator.get_hedgehog_prompt()
        hedgehog_ids = beam_search_decode(
            generator, hedgehog_prompt, num_beams, length_penalty
        )
        hedgehog_text = generator.decode_output(hedgehog_ids)
        logger.info(f"\nBeams: {num_beams}, Length Penalty: {length_penalty}")
        logger.info(hedgehog_text)
        logger.info("\n" + "="*50 + "\n")

    logger.info("Generating JSON outputs with different parameters:")
    for num_beams, length_penalty in parameters:
        json_prompt = generator.get_json_prompt()
        json_ids = beam_search_decode(
            generator, json_prompt, num_beams, length_penalty
        )
        json_text = generator.decode_output(json_ids)
        logger.info(f"\nBeams: {num_beams}, Length Penalty: {length_penalty}")
        logger.info(json_text)
        logger.info("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
