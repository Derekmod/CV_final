import evaluation
import argparse
from model_spec import ModelSpec
import os


parser = argparse.ArgumentParser()

parser.add_argument("--step", type=str)

args = parser.parse_args()

spec = ModelSpec(False, step=args.step, parent_dir='../frame_model_long')

evaluation.copyModel(spec, '../tempModel')
os.system("python2 ../youtube-8m/eval.py --eval_data_pattern='../features/train*.tfrecord' --train_dir=../tempModel --run_once=True --model=LogisticModel")
