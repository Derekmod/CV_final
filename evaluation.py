from model_spec import ModelSpec
import os
import math


TEMP_FILENAME = '__temp_predictions.csv'

def evaluateLabel(spec, record_dir, ground_truth_filename, label=None):
    truths = getGroundTruths(ground_truth_filename)

    sum_err = 0.
    for record_name in os.listdir(record_dir):
        generatePredictions(spec, record_name)
        predictions = parsePredictions(TEMP_FILENAME)
        for id in predictions:
            label_list = [label]
            if label is None:
                label_list = predictions[id].keys()

            for tlabel in label_list:
                target = 0
                if label in truths[id]:
                    target = 1

                pred = predictions[id][label]
                sum_err += BCELoss(pred, target)

    return sum_err


def getStepList(spec):
    model_files = os.listdir(spec.parent_dir)
    step_list = []
    for filename in model_files:
        if filename[-4:0] == 'meta':
            step_list += int(filename[11:-5])
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
    os.system('mkdir {}'.format(os.path.join(output_dir, 'export')))
    format_string = 'cp {} {} -r'
    base_string = format_string.format('{}', output_dir)

    file_template = 'model.ckpt-{}*'.format(spec.step)
    file_template = os.path.join(spec.parent_dir, file_template)
    os.system(base_string.format(file_template))

    export_template = 'export/step_{}/*'.format(spec.step)
    export_template = os.path.join(spec.parent_dir, export_template)
    os.system(base_string.format(export_template))


def generatePredictions(spec, record_name):
    model_dir = 't-' + spec.parent_dir
    copyModel(spec, model_dir)

    format_string = 'python inference.py --outfule_file={} --input_data_pattern={} --train_dir={}'
    command = format_string.format(TEMP_FILENAME, record_name, model_dir)
    os.system(command)

def getGroundTruths(filename):
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
        id, data = line.strip().split(',')
        labels = data.split(' ')
        truths[id] = set([int(label) for label in labels])

    return truths


def parsePredictions(filename):
    predictions = dict()

    file = open(filename)
    file.readline()
    for line in file:
        id, data = line.strip().split(',')
        items = data.split(' ')
        label_probs = dict()
        for idx in range(0, len(data), 2):
            label_probs[data[idx]] = float(data[idx+1])
        predictions[id] = label_probs
    return predictions





def BCELoss(pred, target):
    if type(target) is bool:
        target = int(target)
    target = float(target)

    return - target * math.log(1.-pred) - (1-target) * math.log(pred)