import argparse
from functools import partial
import logging
import os
import pickle

import datasets
import evaluate
import mlflow
import numpy as np
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


# def compute_bleu_score(preds):
#     logits = preds.predictions[0]
#     preds_tok = np.argmax(logits, axis=1)
#     acts = preds.label_ids

#     decode_predictions = tokenizer.batch_decode(preds_tok,
#                                                 skip_special_tokens=True)
#     decode_labels = tokenizer.batch_decode(acts, skip_special_tokens=True)

#     res = bleu.compute(predictions=decode_predictions, references=decode_labels)
#     return {'bleu_score': res['bleu']}


def compute_chrf_score(preds):
    logits = preds.predictions[0]
    preds_tok = np.argmax(logits, axis=1)
    acts = preds.label_ids

    decode_predictions = tokenizer.batch_decode(preds_tok,
                                                skip_special_tokens=True)
    decode_labels = tokenizer.batch_decode(acts, skip_special_tokens=True)

    res = chrf.compute(predictions=decode_predictions, references=decode_labels)
    return {'chrf_score': res['score']}


def main(args):
    checkpoint = args.checkpoint
    data_path = args.data_path
    batch_size = args.batch_size
    seq_length = args.seq_length
    epochs = args.epochs
    model_dir = args.model_dir

    handler = logging.StreamHandler()
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info('Load data')
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info('Load tokenizer and model from HF')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    if 'deci' in str.lower(checkpoint):
        FIM_PREFIX = "<fim_prefix>"
        FIM_MIDDLE = "<fim_middle>"
        FIM_SUFFIX = "<fim_suffix>"
        FIM_PAD = "<fim_pad>"
        EOD = "<|endoftext|>"

        tokenizer.add_special_tokens({
            "additional_special_tokens": [
                EOD, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD
            ],
            "pad_token": EOD,
        })
    else:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, trust_remote_code=True).to(device)

    logger.info('Tokenize data')
    tokenize_partial = partial(tokenize_function,
                               tokenizer=tokenizer,
                               seq_length=seq_length)

    tokenized_dataset = data.map(tokenize_partial,
                                 batched=True,
                                 batch_size=batch_size).map(
                                     add_labels,
                                     batched=True,
                                     batch_size=batch_size)

    # Data collator - Assembles data into batches for training
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    #training_args = TrainingArguments(output_dir='./output',
    training_args = TrainingArguments(output_dir=model_dir,
                                      gradient_checkpointing=True,
                                      evaluation_strategy="epoch",
                                      num_train_epochs=epochs)

    #bleu = evaluate.load("bleu")
    chrf = evaluate.load("chrf")

    logger.info('Finetune model')
    with mlflow.start_run():
        mlflow.transformers.autolog(log_models=False)
        trainer = Trainer(model,
                          training_args,
                          train_dataset=tokenized_dataset['train'],
                          eval_dataset=tokenized_dataset['test'],
                          data_collator=data_collator,
                          compute_metrics=compute_chrf_score,
                          tokenizer=tokenizer)

    trainer.train()

    trainer.save_model()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--seq_length", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--model_dir", type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    main(args)
