import argparse
from functools import partial
import logging
import os
import pickle

import datasets
#import evaluate
import mlflow
import numpy as np
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

import funcs


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
    fp16_flag = True if torch.cuda.is_available() else False

    logger.info('Load tokenizer and model from HF')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        trust_remote_code=True).to(device)

    logger.info('Prepare model for training')
    model.train()
    model = prepare_model_for_int8_training(model)

    logger.info('Tokenize data')
    tokenize_partial = partial(funcs.tokenize_function,
                               tokenizer=tokenizer,
                               seq_length=seq_length)

    tokenized_dataset = data.map(tokenize_partial,
                                 batched=True,
                                 batch_size=batch_size).map(
                                     funcs.add_labels,
                                     batched=True,
                                     batch_size=batch_size)

    # Data collator - Assembles data into batches for training
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)

    training_args = TrainingArguments(
        output_dir=model_dir,
        gradient_checkpointing=True,
        learning_rate=3e-4,
        fp16=True,
        evaluation_strategy="steps",  # if val_set_size > 0 else "no",
        num_train_epochs=epochs,
    )

    compute_metrics_partial = partial(funcs.compute_metrics,
                                      tokenizer=tokenizer,
                                      checkpoint=checkpoint)

    logger.info('Finetune model')
    with mlflow.start_run():
        mlflow.transformers.autolog(log_models=False)
        trainer = Trainer(model,
                          training_args,
                          train_dataset=tokenized_dataset['train'].select(range(10)),
                          eval_dataset=tokenized_dataset['test'].select(range(3)),
                          data_collator=data_collator,
                          compute_metrics=compute_metrics_partial,
                          tokenizer=tokenizer)
        model.config.use_cache = False

        old_state_dict = model.state_dict
            model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
                model, type(model)
            )
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        trainer.train()

        # this saves the peft adapters
        trainer.model.save_pretrained(output_dir)


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
