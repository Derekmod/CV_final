from model_spec import ModelSpec
import os
import math


TEMP_FILENAME = '__temp_predictions.csv'

def evaluateLabel(spec, record_dir, truths, label=None, output_filename=None, train=True, num_records=10):
    #truths = getGroundTruths(ground_truth_filename)
    label_list = [label]
    if label is None:
        label_list = range(1000)

    sum_err = 0.
    label_errs = dict()
    trial = 0
    print record_dir
    for record_name in getTFRecords(record_dir, train)[:num_records]:
        print 'reading record %d' % trial
        trial += 1
        generatePredictions(spec, record_name)
        predictions = parsePredictions(TEMP_FILENAME)
        for vid in predictions:

            for tlabel in label_list:
                target = 0
                if tlabel in truths[vid]:
                    target = 1

                if tlabel not in predictions[vid]:
                    pred = 0.005
                else:
                    pred = predictions[vid][tlabel]
                err = BCELoss(pred, target)

                sum_err += err
                if tlabel not in label_errs:
                    label_errs[tlabel] = 0.
                label_errs[tlabel] += err

    if output_filename is not None:
        outfile = open(output_filename, 'w')
        for tlabel in label_errs:
            outfile.write('%d,%f\n' % (tlabel, label_errs[tlabel]))

    if label is None:
        return label_errs
    return sum_err


def getStepList(spec):
    model_files = os.listdir(spec.parent_dir)
    step_list = []
    for filename in model_files:
        if filename[-4:] == 'meta':
            step_list += [int(filename[11:-5])]
    if 0 in step_list:
        step_list.remove(0)
    return step_list


def stepPredictions(spec, record_name, video_id, output_filename):
    outfile = open(output_filename, 'w')
    outfile.write('Step,Predictions\n')

    step_list = getStepList(spec)

    for step in step_list:
        generatePredictions(spec, record_name)
        predictions = parsePredictions(TEMP_FILENAME)

        predictions_list = []
        for label in predictions:
            predictions_list += [str(label) + ' ' + str(predictions[label])]

        outfile.write('{},{}\n'.format(step, ' '.join(predictions_list)))

def copyModel(spec, output_dir):
    os.system('rm -r {}'.format(output_dir))
    os.system('mkdir {}'.format(output_dir))
    export_dir = os.path.join(output_dir, 'export')
    os.system('mkdir {}'.format(export_dir))
    step_dir = os.path.join(export_dir, 'step_{}'.format(spec.step))
    #os.system('mkdir {}'.format(step_dir))

    src = os.path.join(spec.parent_dir, 'model.ckpt-{}.*'.format(spec.step))
    src2 = os.path.join(spec.parent_dir, 'model.ckpt-{}.index'.format(spec.step))
    src3 = os.path.join(spec.parent_dir, 'model.ckpt-{}.meta'.format(spec.step))
    dest = output_dir
    os.system('cp {} {}'.format(src, dest))
    os.system('cp {} {}'.format(src2, dest))
    os.system('cp {} {}'.format(src3, dest))

    step_ext = 'export/step_{}'.format(spec.step)
    src = os.path.join(spec.parent_dir, step_ext)
    dest = os.path.join(output_dir, 'export')
    os.system('cp -r {} {}'.format(src, dest))

    f = open(os.path.join(output_dir, 'checkpoint'), 'w')
    f.write('model_checkpoint_path: "model.ckpt-{}"\n'.format(spec.step))


def generatePredictions(spec, record_name):
    if spec.step < 0:
        model_dir = spec.parent_dir
    else:
        model_dir = spec.parent_dir + "t"
        copyModel(spec, model_dir)
    print record_name + " generatePredictions"

    format_string = 'python ../youtube-8m/inference.py --output_file={} --input_data_pattern={} --train_dir={}'
    command = format_string.format(TEMP_FILENAME, record_name, model_dir)
    #pid = os.system(command)
    print 'status of "%s" = %d' % (command, os.system(command))
    #os.waitpid(pid, 0)

def getGroundTruths(filename): #verified
    '''
    Args:
        filename: name of ground truths file
    Returns:
        truths: video_id --> set<int label>
    '''
    truths = dict()

    file = open(filename)
    #file.readline()
    for line in file:
        if len(line) < 10:
            continue
        vid, data = line.strip().split(',')
        labels = data.split(' ')
        truths[vid] = set([int(label) for label in labels])

    return truths


def parsePredictions(filename):
    predictions = dict()

    file = open(filename)
    file.readline()
    for line in file:
        vid, data = line.strip().split(',')
        items = data.split(' ')
        label_probs = dict()
        for idx in range(0, len(items), 2):
            label_probs[items[idx]] = float(items[idx+1])
        predictions[vid] = label_probs
    return predictions





def BCELoss(pred, target):
    if type(target) is bool:
        target = int(target)
    target = float(target)

    return - target * math.log(pred) - (1-target) * math.log(1.-pred)

def getTFRecords(feature_dir, isTrain):
    res = []
    print feature_dir + " in getTFRecords"
    for filename in os.listdir(feature_dir):
        if isTrain:
            if filename.startswith("train"):
                res.append(os.path.join(feature_dir,filename))
        else:
            if filename.startswith("validate"):
                res.append(os.path.join(feature_dir,filename))
    return res
