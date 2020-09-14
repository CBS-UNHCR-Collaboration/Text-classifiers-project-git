"""
Raghava Rao Mukkamala (rrm.digi@cbs.dk)
Utilities class for classifiers

"""
import datetime
import numpy as np
import os


def GetFilenameWithoutExtension(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


def GetCurrentTimestamp():
    return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S '))



def ComputePerformanceMeasuresForMultiLabelClassification(actual_labels, predicted_labels):

    result = 'Exact match Ratio: ' + str(ExactMatchRatioMultiLabel(actual_labels, predicted_labels)) + '\n'
    result += 'Multi Label Precision :' + str(PrecisionMultiLabel(actual_labels, predicted_labels)) + '\n'
    result += 'Multi Label Recall :' + str(RecallMultiLabel(actual_labels, predicted_labels)) + '\n'
    result += 'Multi Label Accuracy :' + str(AccuracyMultiLabel(actual_labels, predicted_labels)) + '\n'
    result += 'Multi Label F1-measure :' + str(F1_MeasureMultiLabel(actual_labels, predicted_labels)) + '\n'

    return result

def ExactMatchRatioMultiLabel(actual_labels, predicted_labels):

    if np.array(actual_labels).shape != np.array(predicted_labels).shape:
        raise ValueError('shapes of given arrays is not equal!')

    total_observations = len(actual_labels)

    zipped_observations = zip(actual_labels, predicted_labels)

    match_observations = 0

    for (act, pre) in zipped_observations:
        if set(act) == set(pre):
            match_observations += 1

    return match_observations / total_observations


def AccuracyMultiLabel(actual_labels, predicted_labels):
    if np.array(actual_labels).shape != np.array(predicted_labels).shape:
        raise ValueError('shapes of given arrays is not equal!')

    total_observations = len(actual_labels)

    zipped_observations = zip(actual_labels, predicted_labels)

    accuracy_score = 0

    for (act, pre) in zipped_observations:
        accuracy_score += len(set(act).intersection(set(pre))) / len(set(act).union(set(pre)))

    return accuracy_score / total_observations


def PrecisionMultiLabel(actual_labels, predicted_labels):
    if np.array(actual_labels).shape != np.array(predicted_labels).shape:
        raise ValueError('shapes of given arrays is not equal!')

    total_observations = len(actual_labels)

    zipped_observations = zip(actual_labels, predicted_labels)

    precision_score = 0

    for (act, pre) in zipped_observations:
        if len(set(pre)) == 0:
            continue
        precision_score += len(set(act).intersection(set(pre))) / len(set(pre))

    return precision_score / total_observations


def RecallMultiLabel(actual_labels, predicted_labels):
    if np.array(actual_labels).shape != np.array(predicted_labels).shape:
        raise ValueError('shapes of given arrays is not equal!')

    total_observations = len(actual_labels)

    zipped_observations = zip(actual_labels, predicted_labels)

    recall_score = 0

    for (act, pre) in zipped_observations:
        recall_score += len(set(act).intersection(set(pre))) / len(set(act))

    return recall_score / total_observations


def F1_MeasureMultiLabel(actual_labels, predicted_labels):
    if np.array(actual_labels).shape != np.array(predicted_labels).shape:
        raise ValueError('shapes of given arrays is not equal!')

    total_observations = len(actual_labels)

    zipped_observations = zip(actual_labels, predicted_labels)

    f_score = 0

    for (act, pre) in zipped_observations:
        f_score += (2 * len(set(act).intersection(set(pre)))) / (len(set(act)) + len(set(pre)))

    return f_score / total_observations



# def get_most_informative_features(vectorizer, classifier, label_names,max_number_informative_features = 20):
#     """
#     Prints features with the highest coefficient values, per class
#     """
#     feature_names = vectorizer.get_feature_names()
#     output = []
#     for index in range(len(label_names)):
#         output.append('\n' + label_names[index] + ':\n')
#         coefs_with_fns = sorted(zip(classifier.coef_[index], feature_names))
#         top = zip(coefs_with_fns[:max_number_informative_features], coefs_with_fns[:-(max_number_informative_features + 1):-1])
#         for (coef_1, fn_1), (coef_2, fn_2) in top:
#             feat = "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)
#             output.append(feat)
#
#     return '\n'.join(output)
