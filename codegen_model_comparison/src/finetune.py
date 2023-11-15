import argparse
from functools import partial
import logging
import pickle

import datasets
import evaluate
import mlflow
import numpy as np
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)


def tokenize_function(example, tokenizer, batch_size, seq_length):
    return tokenizer(
        example['text'],
        padding="max_length",
        truncation=True,
        batch_size=batch_size,
        max_length=seq_length,
    )


# def add_labels(example):
#     example['label'] = example['input_ids']
#     return example

# def compute_bleu_score(preds):
#     logits = preds.predictions[0]
#     preds_tok = np.argmax(logits, axis=2)
#     acts = preds.label_ids

#     decode_predictions = tokenizer.batch_decode(preds_tok,
#                                                 skip_special_tokens=True)
#     decode_labels = tokenizer.batch_decode(acts, skip_special_tokens=True)

#     res = bleu.compute(predictions=decode_predictions, references=decode_labels)
#     return {'bleu_score': res['bleu']}


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
    tokenize_partial = partial(tokenize_function,
                               tokenizer=tokenizer,
                               batch_size=batch_size,
                               seq_length=seq_length)
    tokenized_dataset = data.map(
        tokenize_partial,
        batched=True
    )  #.map(add_labels, batched=True, batch_size=batch_size)

    # Data collator - Assembles data into batches for training
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(output_dir='trainer_' + checkpoint,
                                      gradient_checkpointing=True,
                                      evaluation_strategy="epoch",
                                      num_train_epochs=1)

    bleu = evaluate.load("bleu")

    logger.info('Finetune model')
    with mlflow.start_run():
        mlflow.transformers.autolog(log_models=False)
        trainer = Trainer(
            model,
            training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['test'],
            data_collator=data_collator,
            #compute_metrics=compute_bleu_score,
            tokenizer=tokenizer)

    trainer.train()

    trainer.save_model()


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
