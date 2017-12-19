import sys, os, argparse
from evaluation import *

# parser = argparse.ArgumentParser()
# parser.add_argument("--feature_dir", type=str)
# parser.add_argument("--ground_truth", type=str)
# parser.add_argument("--isTrain", type=bool)
# args = parser.parse_args()
# if len(args) != 3:
#     print "USAGE: python labelCount.py --feature_dir=[] --ground_truth=[] --train=[]"
# else:
#     gtDict = getGroundTruths(args.ground_truth)
#     counts = dict()
#     for filename in os.listdir(args.feature_dir):
#         if args.isTrain:
#             if "train" not in filename:
#                 continue
#             f = open(args.feature_dir + "/" + filename).read()
#             vIds = f.split("video_id")
#             for vid in vIds:
#                 labelSet = gtDict.get(vid[6:17])


def labelCount(feature_dir, gtDict, isTrain):
    #gtDict = getGroundTruths(ground_truth)
    counts = dict()
    counts[""] = 0
    for filename in getTFRecords(feature_dir, isTrain):
        f = open(filename).read()
        vIds = f.split("video_id")
        vIds = vIds[1:]
        for vid in vIds:
            counts[""] = counts[""] + 1
            labelSet = gtDict.get(vid[6:17])
            for label in labelSet:
                if label not in counts:
                    counts[label] = 1
                else:
                    counts[label] = counts[label] + 1
    return counts
