"""
Raghava Rao Mukkamala (rrm.digi@cbs.dk)
Base Python class for different Classifiers.
dated: 2017-07-22

"""

import sys
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from ClassifierUtilities import *



class ClassifierBase:

    grid_parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                       'tfidf__use_idf': (True, False),
                       'clf__alpha': (1e-2, 1e-3),
                       }
    vectorizer_options = 'decode_error=\'ignore\', stop_words=\'english\', max_features=10000'

    max_number_informative_features = 20


    def __init__(self, docs_train, y_train, category_labels, debug_mode):
        self.train_docs = docs_train
        self.training_labels = y_train
        self.label_names = category_labels
        self.test_docs = None
        self.debug_mode = debug_mode
        self.pipe_clf = None
        self.text_clf = None
        self.gs_clf = None
        self.gs_test_predict_lables = None
        self.test_predict_lables = None


        return


    def _train(self):

        return

    def _predict(self, docs_test):
        self.test_docs = docs_test
        return

    def _grid_train(self):

        return

    def grid_predict(self, doc_test):
        self.test_docs = doc_test



    def get_most_informative_features(self):
        """
        Prints features with the highest coefficient values, per class
        """
        output = []

        try:

            vectorizer = self.pipe_clf.named_steps['vect']
            clf = self.pipe_clf.named_steps['clf']
            feature_names = vectorizer.get_feature_names()
            #print(feature_names)

            for index in range(len(self.label_names)):
                output.append('\n' + self.label_names[index] + ':\n')
                #print(output)
                coefs_with_fns = sorted(zip(clf.coef_[index], feature_names))
                top = zip(coefs_with_fns[:self.max_number_informative_features],
                          coefs_with_fns[:-(self.max_number_informative_features + 1):-1])
                for (coef_1, fn_1), (coef_2, fn_2) in top:
                    feat = "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)
                    output.append(feat)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            return '\n'.join(output)

        return '\n'.join(output)

    def preformance_measures(self, actual_labels):

        if self.test_predict_lables is None:
            raise ValueError('You have to call train and predict before calling this function')

        measures_info = metrics.classification_report(actual_labels, self.test_predict_lables,
                                                      target_names=self.label_names)

        return measures_info

    def grid_preformance_measures(self, actual_labels):

        if self.gs_test_predict_lables is None:
            raise ValueError('You have to call grid_train and gris_predict before calling this function!')

        measures_info = metrics.classification_report(actual_labels, self.gs_test_predict_lables,
                                                      target_names=self.label_names)

        return measures_info




