import argparse
import logging
import pickle
import os
import sys

import datasets
import evaluate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset_test import prompt, answer
import funcs


def main(args):
    checkpoint = args.checkpoint
    model_folder = args.model_folder
    output_dir = args.output_dir

    handler = logging.StreamHandler()
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info('Load tokenizer and model from HF')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, trust_remote_code=True).to(device)

    logger.info('Predict baselines')
    baseline_predictions = list(
        map(
            lambda text: funcs.generate_text(
                text,
                model,
                tokenizer,
                device,
                repetition_penalty=50.0,
                pad_token_id=tokenizer.pad_token_id,
                #max_new_tokens=1000,
                #min_length=200,  # Commenting out for CodeParrot, otherwise get IndexError
                max_length=1000  # for CodeParrot only
            ),
            prompt))

    with open(output_dir + '/preds_baseline.pkl', 'wb') as f:
        pickle.dump(baseline_predictions, f)

    bleu = evaluate.load("bleu")
    bleu_results = bleu.compute(predictions=baseline_predictions,
                                references=answer)
    chrf = evaluate.load("chrf")
    chrf_results = chrf.compute(predictions=baseline_predictions,
                                references=answer)
    metrics_baseline = {'bleu': bleu_results, 'chrf': chrf_results}

    with open(output_dir + '/metrics_baseline.pkl', 'wb') as f:
        pickle.dump(metrics_baseline, f)

    logger.info('Load finetuned model')
    model_finetuned = AutoModelForCausalLM.from_pretrained(model_folder).to(
        device)

    logger.info('Predict on test data')
    test_predictions = list(
        map(
            lambda text: funcs.generate_text(
                text,
                model_finetuned,
                tokenizer,
                device,
                repetition_penalty=50.0,
                pad_token_id=tokenizer.pad_token_id,
                #min_length=200, # commenting out for CodeParrot, otherwise get IndexError
                max_length=1000)  # for CodeParrot
            ,
            prompt))

    with open(output_dir + '/preds_test.pkl', 'wb') as f:
        pickle.dump(test_predictions, f)

    logger.info('Calculate metrics on finetuned data')
    bleu_results = bleu.compute(predictions=test_predictions, references=answer)
    chrf_results = chrf.compute(predictions=test_predictions, references=answer)
    metrics_test = {'bleu': bleu_results, 'chrf': chrf_results}

    with open(output_dir + '/metrics_test.pkl', 'wb') as f:
        pickle.dump(metrics_test, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--model_folder", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()
    main(args)
