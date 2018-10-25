import data_helper
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import math


PREDICTOR_COL = [i for i in range(0, 10)]
LABEL_COL = 10


def cross_entropy(mod_, data_, cols, auc=False):
    # Predict class (gamma or hadron)
    pred_ = mod_.predict_proba(data_[:, cols])
    # Prediction of class gamma is in second column.
    term_1 = data_[:, LABEL_COL] * np.log2(pred_[:, 1])
    term_2 = (1 - data_[:, LABEL_COL]) * np.log2(1 - pred_[:, 1])
    add_term = term_1 + term_2
    neg_avg = -(np.sum(add_term) / add_term.shape[0])
    if auc:
        a_score = roc_auc_score(data_[:, LABEL_COL], pred_[:, 1])
        return neg_avg, a_score
    return neg_avg


def entropy_auc(training, validation, testing, features):
    print('Final feature selection: {}'.format(features))
    # Confirm cross entropy for validation data for x_sel
    model_ = MLPClassifier()
    model_.fit(training[:, features], training[:, LABEL_COL])
    cros_entrpy_valida_, auc_validation = cross_entropy(model_, validation, features, True)
    print('Cross entropy for validation data with '
          'selected features is : {}'
          .format(cros_entrpy_valida_))
    print('AUC for validation data with '
          'selected features is : {}'
          .format(auc_validation))

    # Get the cross entropy for testing data for x_sel
    cros_entrpy_test_, auc_testing = cross_entropy(model_, testing, features, True)
    print('Cross entropy for testing data with '
          'selected features is : {}'
          .format(cros_entrpy_test_))
    print('AUC for validation data with '
          'selected features is : {}'
          .format(auc_testing))



def feature_sel(data_):
    train_, valid_, test_ = data_helper.split_data(data_)
    x_sel = [i for i in range(0, 10)]
    # Initial cross entropy.
    model_ = MLPClassifier()
    model_.fit(train_[:, x_sel], train_[:, LABEL_COL])
    prev_min_cros_entrpy = 100
    min_cros_entrpy = cross_entropy(model_, valid_, x_sel)
    while (prev_min_cros_entrpy + (0.01 * prev_min_cros_entrpy)) > min_cros_entrpy:
        prev_min_cros_entrpy = min_cros_entrpy
        crs_entr_li = list()
        for feat in x_sel:
            train_features = [x for x in x_sel if x != feat]
            model_ = MLPClassifier()
            print('Training the model with feature(s): {}'.format(train_features))
            model_.fit(train_[:, train_features], train_[:, LABEL_COL])
            # print(model_.classes_)
            print('Calculating entropy')
            crs_entr_li.append(cross_entropy(model_, valid_, train_features))
        print('Entropies are: {}'.format(crs_entr_li))
        print('Minimum entropy is {} and is obtained for feature: {}'
              .format(min(crs_entr_li), x_sel[crs_entr_li.index(min(crs_entr_li))]))
        min_cros_entrpy = min(crs_entr_li)
        x_sel.pop(crs_entr_li.index(min(crs_entr_li)))
        print('Remaining features in this iteration: {}'.format(x_sel))
        print('Previous Entropy: {}, Current Entropy: {}'.format(prev_min_cros_entrpy, min_cros_entrpy))

    # Get output for the problem;
    entropy_auc(train_, valid_, test_, x_sel)


def main():
    """
    This function is the entry point.
    :return:
    """
    data = data_helper.get_file_data()
    feature_sel(data)


if __name__ == '__main__':
    main()

