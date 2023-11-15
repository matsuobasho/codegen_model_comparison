import argparse
import pickle

import datasets

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