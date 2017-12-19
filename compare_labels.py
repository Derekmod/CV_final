import evaluation
import labelCount
from model_spec import ModelSpec

import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")
parser.add_argument("--model_dir", type=str)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--ground_truths", type=str)
parser.add_argument("-t", "--train", action="store_true")
parser.add_argument("-f", "--frame", action="store_true")
#parser.add_argument("-a", "--augment", action="store_true",
#                    help="augment training data")
#parser.add_argument("--learning_rate", type=float)
#parser.add_argument("--margin", type=float)
#parser.add_argument("--load", type=str, help="file to load weights from")
#parser.add_argument("--save", help="file to save weights to", type=str)
#parser.add_argument("--nepochs", type=int, help="max # of epochs for training")
#parser.add_argument("--data", type=str, help="file to load dataset from")
args = parser.parse_args()


def main():
    # load label counts
    label_counts = labelCount.labelCount(args.data_dir, args.ground_truths, args.train)

    # generate naive errors
    video_count = label_counts['']
    naive_errs = dict()
    for label in label_counts:
        p = float(label_counts[label])/video_count
        naive_errs[label] = evaluation.BCELoss(p, p) * video_count

    # load label errors
    label_errs = evaluation.evaluateLabel(spec, args.data_dir, args.ground_truths)
    pass



if __name__ == '__main__':
    main()