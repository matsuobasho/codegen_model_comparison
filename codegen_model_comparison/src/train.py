import pickle
import sys
from statistics import mean

import argparse
import numpy as np

import datasets
import evaluate
import mlflow
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, AutoConfig


def tokenize_function(example):
    return tokenizer(example['text'],
                     padding="max_length",
                     truncation=True,
                     max_length=seq_length)


def add_labels(example):
    example['label'] = example['input_ids']
    return example


def compute_bleu_score(preds):
    logits = preds.predictions[0]
    # logits is 3 dim with dims bs, seq length, vocab size
    preds_tok = np.argmax(logits, axis=2)
    acts = preds.label_ids

    decode_predictions = tokenizer.batch_decode(preds_tok,
                                                skip_special_tokens=True)
    decode_labels = tokenizer.batch_decode(acts, skip_special_tokens=True)

    res = bleu.compute(predictions=decode_predictions, references=decode_labels)
    return {'bleu_score': res['bleu']}


def main(args):
    data_path = args.data_path  ###!!! to check later
    output_path = args.model_dir
    batch_size = args.batch_size
    #seq_length = args.seq_length
    seq_length = 100
    #epochs = args.epochs
    learning_rate = args.learning_rate

    # Load data
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    checkpoint = "Salesforce/codegen-350M-mono"
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, trust_remote_code=True).to(device)

    tokenized_dataset = data.map(tokenize_function,
                                 batched=True,
                                 batch_size=batch_size).map(
                                     add_labels,
                                     batched=True,
                                     batch_size=batch_size)

    bleu = evaluate.load("bleu")

    with mlflow.start_run():
        mlflow.transformers.autolog(log_models=False)
        mlflow.set_tags({'token_length': seq_length, 'model': 'codegen'})

        params = {"token_length": seq_length, "model": checkpoint}
        mlflow.log_params(params)

        # Data collator - Assembles data into batches for training
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        training_args = TrainingArguments(
            output_dir=output_path,
            evaluation_strategy="epoch",
            gradient_checkpointing=True,
            #num_train_epochs=epochs,
            learning_rate=learning_rate
            #gradient_accumulation_steps=8,
            #fp16 = True
        )

        trainer = Trainer(model,
                          training_args,
                          train_dataset=tokenized_dataset["train"],
                          eval_dataset=tokenized_dataset["test"],
                          data_collator=data_collator,
                          compute_metrics=compute_bleu_score,
                          tokenizer=tokenizer)

        trainer.train()

        trainer.save_model()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--output_path")
    parser.add_argument("--batch_size", type=int)
    #parser.add_argument("--seq_length", type=int)
    #parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning_rate", type=float)
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    main(args)