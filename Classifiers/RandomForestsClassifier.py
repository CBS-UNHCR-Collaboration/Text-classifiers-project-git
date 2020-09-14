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
from sklearn.ensemble import RandomForestClassifier




class RandomForestsClassifier(ClassifierBase):

    classifier_name = 'RandomForestsClassifier'

    def __init__(self, docs_train, y_train, category_labels, debug_mode):
        super().__init__(docs_train, y_train, category_labels, debug_mode)

        self.grid_parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                                'tfidf__use_idf': (True, False),
                                #'clf__penalty': ('l1', 'l2'),
                                'clf__bootstrap': (True, False),
                                }

        return

    def train(self):

        super()._train()

        if self.debug_mode:
            print(GetCurrentTimestamp() + ': Start of RandomForests training!')
        self.pipe_clf = Pipeline(
            [('vect', CountVectorizer(decode_error='ignore', stop_words='english', max_features=10000)),
             ('tfidf', TfidfTransformer()),
             ('clf', RandomForestClassifier(n_estimators=100)),
             ])
        self.text_clf = self.pipe_clf.fit(self.train_docs, self.training_labels)

        if self.debug_mode:
            print(GetCurrentTimestamp() + ': End of RandomForests training!')

        return self.text_clf

    def predict(self, doc_test):

        super()._predict(doc_test)

        if self.debug_mode:
            print(GetCurrentTimestamp() + ': Start of RandomForests Predict!')

        if self.text_clf is None:
            self.train()

        self.test_predict_lables = self.text_clf.predict(doc_test)
        if self.debug_mode:
            print(GetCurrentTimestamp() + ': End of RandomForests Predict!')

        return self.test_predict_lables

    def grid_train(self):
        if self.debug_mode:
            print(GetCurrentTimestamp() + ': Start of RandomForests Grid Search!')

        if self.text_clf is None:
            self.train()

        self.gs_clf = GridSearchCV(self.text_clf, self.grid_parameters, n_jobs=-1)

        gs_clf = self.gs_clf.fit(self.train_docs, self.training_labels)

        if self.debug_mode:
            print(GetCurrentTimestamp() + ': End of RandomForests Grid Search!')

        return self.gs_clf

    def grid_predict(self, doc_test):
        if self.debug_mode:
            print(GetCurrentTimestamp() + ': Start of RandomForests Grid Predict!')

        self.gs_test_predict_lables = self.gs_clf.predict(doc_test)
        if self.debug_mode:
            print(GetCurrentTimestamp() + ': End of RandomForests Grid Predict!')

        return self.gs_test_predict_lables
