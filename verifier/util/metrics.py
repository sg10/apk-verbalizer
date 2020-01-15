# has been removed from keras in v2,
# taken from https://github.com/keras-team/keras/issues/5400

from keras import backend as K
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.contrib.framework import argsort

from verifier.util.inspect import copy_func
from verifier.util.string_table_builder import StringTable

# set in a way so that the median number of values per samples larger than the threshold is 10
_tfidf_discretize_threshold_descriptions = 0.14


def tfidf_accuracy(y_true, y_pred, threshold=_tfidf_discretize_threshold_descriptions):
    y_true_gt0_mask = tf.greater(y_true, threshold)
    y_pred_gt0_mask = tf.greater(y_pred, threshold)

    and_mask = tf.logical_and(y_true_gt0_mask, y_pred_gt0_mask)

    ones_correct = tf.boolean_mask(tf.ones(tf.shape(y_true)), and_mask)
    num_correct = tf.size(ones_correct)

    ones_true = tf.boolean_mask(tf.ones(tf.shape(y_true)), y_true_gt0_mask)
    num_true = tf.size(ones_true)

    return np.true_divide(num_correct, num_true)


def tfidf_metrics(y_true, y_pred, threshold=_tfidf_discretize_threshold_descriptions):
    y_true_gt0_mask = tf.greater(y_true, threshold)
    y_pred_gt0_mask = tf.greater(y_pred, threshold)

    # reduce both to the elements where either one or the other is > threshold
    mask_both = tf.logical_or(y_true_gt0_mask, y_pred_gt0_mask)

    y_true_relevant = tf.boolean_mask(y_true, mask_both)
    y_pred_relevant = tf.boolean_mask(y_pred, mask_both)

    y_true_relevant_gt0_mask = tf.greater(y_true_relevant, threshold)
    y_pred_relevant_gt0_mask = tf.greater(y_pred_relevant, threshold)

    y_true_relevant_binary = tf.cast(y_true_relevant_gt0_mask, dtype=tf.int32)
    y_pred_relevant_binary = tf.cast(y_pred_relevant_gt0_mask, dtype=tf.int32)

    correct_true_positives = tf.reduce_sum(tf.multiply(y_true_relevant_binary, y_pred_relevant_binary))
    pred_positives = tf.reduce_sum(y_pred_relevant_binary)
    true_positives = tf.reduce_sum(y_true_relevant_binary)

    precision = tf.clip_by_value(np.true_divide(correct_true_positives, pred_positives), 0.0, 1.0)
    recall = np.true_divide(correct_true_positives, true_positives)
    f1_score = 2 * tf.divide(tf.multiply(precision, recall), tf.add(precision, recall))

    return precision, recall, f1_score


def tfidf_precision(y_true, y_pred, threshold=_tfidf_discretize_threshold_descriptions):
    return tfidf_metrics(y_true, y_pred, threshold)[0]


def tfidf_recall(y_true, y_pred, threshold=_tfidf_discretize_threshold_descriptions):
    return tfidf_metrics(y_true, y_pred, threshold)[1]


def tfidf_f1(y_true, y_pred, threshold=_tfidf_discretize_threshold_descriptions):
    return tfidf_metrics(y_true, y_pred, threshold)[2]


def get_tfidf_parameterized(beta, threshold):

    def tfidf_fbeta(y_true, y_pred):
        p, r, _ = tfidf_metrics(y_true, y_pred, threshold)
        fb = (1 + beta * beta) * p * r / ((beta * beta * p) + r + K.epsilon())
        return fb

    def tfidf_precision(y_true, y_pred):
        return tfidf_metrics(y_true, y_pred, threshold)[0]

    def tfidf_recall(y_true, y_pred):
        return tfidf_metrics(y_true, y_pred, threshold)[1]

    th_str = ("%.4f" % threshold).strip("0").strip(".")

    return [copy_func(tfidf_fbeta, "f_%s" % th_str),
            copy_func(tfidf_precision, "pr_%s" % th_str),
            copy_func(tfidf_recall, "rc_%s" % th_str)]


def class_report_fbeta(y_pred, y_true, labels, beta, print_output=False):
    precision, recall, fbeta, support = precision_recall_fscore_support(y_true, y_pred, beta=beta)
    macro_precision, macro_recall, macro_fbeta, macro_support = \
        precision_recall_fscore_support(y_true, y_pred, beta=beta, average='macro')
    micro_precision, micro_recall, micro_fbeta, micro_support = \
        precision_recall_fscore_support(y_true, y_pred, beta=beta, average='micro')

    report = {}

    for i, label in enumerate(labels):
        report[label] = {
            'precision': precision.tolist()[i],
            'recall': recall.tolist()[i],
            'fbeta': fbeta.tolist()[i],
            'support': support.tolist()[i],
        }

    report['_macro'] = {
        'precision': macro_precision,
        'recall': macro_recall,
        'fbeta': macro_fbeta,
        'support': macro_support
    }

    report['_micro'] = {
        'precision': micro_precision,
        'recall': micro_recall,
        'fbeta': micro_fbeta,
        'support': micro_support
    }

    if print_output:
        print_class_report_fbeta(report)

    return report


def print_class_report_fbeta(report):
    tbl = StringTable(separator=" ")
    keys = list(report.keys())
    keys.remove("_macro")
    keys.remove("_micro")
    keys.append("_macro")
    keys.append("_micro")

    cols = ["precision", "recall", "fbeta", "support"]

    for key in keys:
        row = [key]
        for c in cols:
            if report[key][c] is not None:
                if c == "support":
                    row.append("%5d" % report[key][c])
                else:
                    row.append("%.2f" % report[key][c])
        tbl.add_cells(row)
        tbl.new_row()

    cols.insert(0, "")

    tbl.set_headline(cols)

    print(tbl.create_table(False))


if __name__ == "__main__":
    array_y_true = np.array([[0] * 10 + [1, 2, 3, 4] + [0] * 0,
                             [0] * 9 + [1, 2, 3, 4] + [0] * 1,
                             [0] * 9 + [1, 2, 3, 4] + [0] * 1])
    array_y_pred = np.array([[0] * 10 + [1, 2, 3, 4] + [0] * 0,
                             [0] * 10 + [1, 2, 3, 4] + [0] * 0,
                             [0] * 1 + [1, 2, 3, 0] + [0] * 9])

    y_true = tf.placeholder(tf.float32, shape=array_y_true.shape)
    y_pred = tf.placeholder(tf.float32, shape=array_y_pred.shape)

    calc = tfidf_metrics(y_true, y_pred)

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    val = \
    {y_true: array_y_true,
     y_pred: array_y_pred}
    res = sess.run(calc, val)

    print(array_y_pred)
    print(array_y_true)

    print(res[0])
    print(res[1])
    print(res[2])