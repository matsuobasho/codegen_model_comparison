import argparse
import logging
import pickle
import os
import sys

import datasets
import evaluate
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from dataset_test import prompt, answer


def tokenize_function(example, tokenizer, seq_length):
    return tokenizer(
        example['text'],
        padding="max_length",
        truncation=True,
        max_length=seq_length,
    )


def add_labels(example):
    example['label'] = example['input_ids']
    return example


def generate_text(prompt, model_, tok_, device, **kwargs):
    model_inputs = tok_(prompt, return_tensors="pt").to(device)
    generated_ids = model_.generate(**model_inputs, **kwargs)
    # feed only the newly-generated ids to decode
    res = tok_.batch_decode(generated_ids[:,
                                          model_inputs['input_ids'].shape[1]:],
                            skip_special_tokens=True)[0]

    return res


def main(args):
    checkpoint = args.checkpoint
    # test_data_path = args.test_data_path
    # metrics_path = args.metrics_path
    # baseline_preds_path = args.baseline_preds_path
    model_folder = args.model_folder
    output_dir = args.output_dir

    handler = logging.StreamHandler()
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info('Load finetuned model')
    model_finetuned = AutoModelForCausalLM.from_pretrained(model_folder,
                                                           device_map="auto")
    logger.info('Load tokenizer and model from HF')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, trust_remote_code=True).to(device)

    baseline_predictions = list(
        map(
            lambda text: generate_text(text,
                                       model,
                                       tokenizer,
                                       device,
                                       repetition_penalty=50.0,
                                       max_new_tokens=1000), prompt))

    bleu = evaluate.load("bleu")
    bleu_results = bleu.compute(predictions=baseline_predictions,
                                references=answer)
    chrf = evaluate.load("chrf")
    chrf_results = chrf.compute(predictions=baseline_predictions,
                                references=answer)
    metrics = {'bleu': bleu_results, 'chrf': chrf_results}

    logger.info(f'Saving to {os.getcwd()}')
    with open(output_dir + '/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    # with open(output_path + '/baseline_preds.pkl', 'wb') as f:
    #     pickle.dump(baseline_predictions, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--test_data_path", type=str)
    # parser.add_argument("--metrics_path", type=str)
    # parser.add_argument("--baseline_preds_path", type=str)
    parser.add_argument("--model_folder", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    main(args)
