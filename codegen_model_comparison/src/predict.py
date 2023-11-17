import argparse
import pickle

import datasets


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


def main(args):
    data_path = args.data_path
    with open(data_path, "rb") as f:
        data = pickle.load(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    main(args)
