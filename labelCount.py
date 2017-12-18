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


def labelCount(feature_dir, ground_truth, isTrain):
    gtDict = getGroundTruths(ground_truth)
    counts = dict()
    for filename in getTFRecords(feature_dir, isTrain):
        f = open(feature_dir + "/" + filename).read()
        vIds = f.split("video_id")
        for vid in vIds:
            labelSet = gtDict.get(vid[6:17])
            for label in labelSet:
                if label not in counts:
                    counts[label] = 1
                else:
                    counts[label] = counts[label] + 1
    return counts

def getTFRecords(feature_dir, isTrain):
    res = []
    for filename in os.listdir(feature_dir):
        if isTrain:
            if "train" in filename:
                res.append(filename)
        else:
            if "validate" in filename:
                res.append(filename)
    return res
