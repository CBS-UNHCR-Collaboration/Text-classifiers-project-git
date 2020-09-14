"""
Raghava Rao Mukkamala (rrm.digi@cbs.dk)
Derived Python class for Naive Bayes Multiminal Classifier
dated: 2017-07-22

"""

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from ClassifierUtilities import *
from Classifiers.ClassifierBase import ClassifierBase
from sklearn.linear_model import LogisticRegression


class LogisticRegressionClassifier(ClassifierBase):
    classifier_name = 'LogisticRegressionClassifier'

    def __init__(self, docs_train, y_train, category_labels, debug_mode):
        super().__init__(docs_train, y_train, category_labels, debug_mode)

        self.grid_parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                                'tfidf__use_idf': (True, False),
                                'clf__penalty': ('l1', 'l2'),
                                'clf__tol': (1e-1, 1e-3),
                                }

        # self.grid_parameters = [{'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        #                          'tfidf__use_idf': (True, False),
        #                          'clf__penalty': ['l1'],
        #                          'solver': ['lbfgs', 'liblinear', 'sag', 'saga'],
        #                          'clf__tol': (1e-1, 1e-3),
        #                          },
        #                         {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        #                          'tfidf__use_idf': (True, False),
        #                          'clf__penalty': ['l2'], 'solver': ['newton-cg'],
        #                          'clf__tol': (1e-1, 1e-3),
        #                          }
        #                         ]

        return

    def train(self):

        super()._train()

        if self.debug_mode:
            print(GetCurrentTimestamp() + ': Start of LogisticRegression training!')
        self.pipe_clf = Pipeline(
            [('vect', CountVectorizer(decode_error='ignore', stop_words='english', max_features=10000)),
             ('tfidf', TfidfTransformer()),
             ('clf', LogisticRegression(penalty='l2', dual=False, tol=1e-3, solver='liblinear', multi_class='auto')),
             ])
        self.text_clf = self.pipe_clf.fit(self.train_docs, self.training_labels)

        if self.debug_mode:
            print(GetCurrentTimestamp() + ': End of LogisticRegression training!')

        return self.text_clf

    def predict(self, doc_test):

        super()._predict(doc_test)

        if self.debug_mode:
            print(GetCurrentTimestamp() + ': Start of LogisticRegression Predict!')

        if self.text_clf is None:
            self.train()

        self.test_predict_lables = self.text_clf.predict(doc_test)
        if self.debug_mode:
            print(GetCurrentTimestamp() + ': End of LogisticRegression Predict!')

        return self.test_predict_lables

    def grid_train(self):
        if self.debug_mode:
            print(GetCurrentTimestamp() + ': Start of LogisticRegression Grid Search!')

        if self.text_clf is None:
            self.train()

        self.gs_clf = GridSearchCV(self.text_clf, self.grid_parameters, n_jobs=-1)

        gs_clf = self.gs_clf.fit(self.train_docs, self.training_labels)

        if self.debug_mode:
            print(GetCurrentTimestamp() + ': End of LogisticRegression Grid Search!')

        return self.gs_clf

    def grid_predict(self, doc_test):
        if self.debug_mode:
            print(GetCurrentTimestamp() + ': Start of LogisticRegression Grid Predict!')

        self.gs_test_predict_lables = self.gs_clf.predict(doc_test)
        if self.debug_mode:
            print(GetCurrentTimestamp() + ': End of LogisticRegression Grid Predict!')

        return self.gs_test_predict_lables
