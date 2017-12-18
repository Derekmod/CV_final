import evaluation
from model_spec import ModelSpec

import argparse

import matplotlib.pyplot as plt


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
    spec = ModelSpec(args.frame, step=0, parent_dir=args.model_dir)
    step_list = evaluation.getStepList().sort()

    error_list = dict()

    for step in step_list:
        spec.step = step
        error = evaluation.evaluateLabel(spec, args.data_dir, args.ground_truths, label=None)
        error_list += [error]

    #plot
    plt.plot(step_list, error_list)
    plt.show()