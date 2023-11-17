import argparse
import logging
import pickle
import sys

import datasets
import evaluate
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)


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
    test_data_path = args.test_data_path

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

    # !!! Test whether this is working
    sys.path.insert(0, test_data_path)
    from test_funcs import prompt, answer
    logger.info(prompt)
    logger.info(answer)

    # !!! Then modify this to do it an elegant way
    text1_predict_bfinetune = generate_text(prompt[0],
                                            model,
                                            tokenizer,
                                            device,
                                            repetition_penalty=50.0,
                                            max_new_tokens=1000)
    text2_predict_bfinetune = generate_text(prompt[1],
                                            model,
                                            tokenizer,
                                            device,
                                            repetition_penalty=50.0,
                                            max_new_tokens=1000)
    text3_predict_bfinetune = generate_text(prompt[2],
                                            model,
                                            tokenizer,
                                            device,
                                            repetition_penalty=50.0,
                                            max_new_tokens=1000)

    bleu = evaluate.load("bleu")
    bleu_results = bleu.compute(predictions=[
        text1_predict_bfinetune, text2_predict_bfinetune,
        text3_predict_bfinetune
    ],
                                references=answer)

    chrf_results = chrf.compute(predictions=[
        text1_predict_bfinetune, text2_predict_bfinetune,
        text3_predict_bfinetune
    ],
                                references=answer)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--test_data_path", type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    main(args)
