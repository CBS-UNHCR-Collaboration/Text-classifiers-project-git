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
from sklearn.linear_model import SGDClassifier


class SVMSGDClassifier(ClassifierBase):

    classifier_name = 'SVMSGDClassifier'

    def __init__(self, docs_train, y_train, category_labels, debug_mode):
        super().__init__(docs_train, y_train, category_labels, debug_mode)

        self.grid_parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
               'tfidf__use_idf': (True, False),
                'clf__penalty': ('l1', 'l2'),
               'clf__alpha': (1e-2, 1e-3),
 }
        self.max_iter = 1000

        return

    def train(self):

        super()._train()

        if self.debug_mode:
            print(GetCurrentTimestamp() + ': Start of Linear SVC training!')
        self.pipe_clf = Pipeline(
            [('vect', CountVectorizer(decode_error='ignore', stop_words='english', max_features=10000)),
             ('tfidf', TfidfTransformer()),
             ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, max_iter=1000, random_state=42)),
             ])
        self.text_clf = self.pipe_clf.fit(self.train_docs, self.training_labels)

        if self.debug_mode:
            print(GetCurrentTimestamp() + ': End of Linear SVC training!')

        return self.text_clf

    def predict(self, doc_test):

        super()._predict(doc_test)

        if self.debug_mode:
            print(GetCurrentTimestamp() + ': Start of Linear SVC Predict!')

        if self.text_clf is None:
            self.train()

        self.test_predict_lables = self.text_clf.predict(doc_test)
        if self.debug_mode:
            print(GetCurrentTimestamp() + ': End of Linear SVC Predict!')

        return self.test_predict_lables

    def grid_train(self):
        if self.debug_mode:
            print(GetCurrentTimestamp() + ': Start of Linear SVC Grid Search!')

        if self.text_clf is None:
            self.train()

        self.gs_clf = GridSearchCV(self.text_clf, self.grid_parameters, n_jobs=-1)

        gs_clf = self.gs_clf.fit(self.train_docs, self.training_labels)

        if self.debug_mode:
            print(GetCurrentTimestamp() + ': End of Linear SVC Grid Search!')

        return self.gs_clf

    def grid_predict(self, doc_test):
        if self.debug_mode:
            print(GetCurrentTimestamp() + ': Start of Linear SVC Grid Predict!')

        self.gs_test_predict_lables = self.gs_clf.predict(doc_test)
        if self.debug_mode:
            print(GetCurrentTimestamp() + ': End of Linear SVC Grid Predict!')

        return self.gs_test_predict_lables
