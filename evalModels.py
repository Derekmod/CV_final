import evaluation
import argparse
from model_spec import ModelSpec
import os


parser = argparse.ArgumentParser()

parser.add_argument("--step", type=str)

args = parser.parse_args()

spec = ModelSpec(False, step=args.step, parent_dir='../frame_model_long')

evaluation.copyModel(spec, '../tempModel')
os.system("python2 ../youtube-8m/eval.py --eval_data_pattern='../frame-features/validate*.tfrecord' --train_dir=../tempModel --run_once=True --frame_features=True --model=FrameLevelLogisticModel --feature_names=rgb --feature_sizes=1024" )
