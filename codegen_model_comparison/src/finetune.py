import argparse
import logging
import pickle

import datasets
import evaluate
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)


def tokenize_function(example):
    return tokenizer(example['text'],
                     padding="max_length",
                     truncation=True,
                     max_length=500)


def add_labels(example):
    example['label'] = example['input_ids']
    return example


def main(args):
    checkpoint = args.checkpoint
    data_path = args.data_path
    batch_size = args.batch_size
    seq_length = args.seq_length

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
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, trust_remote_code=True).to(device)

    logger.info('Tokenize data')
    tokenized_dataset = data.map(tokenize_function,
                                 batched=True,
                                 batch_size=batch_size).map(
                                     add_labels,
                                     batched=True,
                                     batch_size=batch_size)

    # Data collator - Assembles data into batches for training
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(output_dir='trainer_' + checkpoint,
                                      evaluation_strategy="epoch")

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        data_collator=data_collator,
        #compute_metrics=compute_bleu_score,
        tokenizer=tokenizer)

    trainer.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--seq_length", type=int)
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    main(args)
