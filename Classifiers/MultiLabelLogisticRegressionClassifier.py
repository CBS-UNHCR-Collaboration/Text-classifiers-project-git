import os
import sys

# parentdir = os.path.dirname(__file__)
# sys.path.insert(0, parentdir)
# parenttoparent = os.path.dirname(parentdir)
# sys.path.insert(0, parenttoparent)

import codecs
import csv

from lxml import etree

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from sklearn.model_selection import train_test_split

from ClassifierUtilities import *


class MultiLabelLogisticRegressionClassifier:



    def __init__(self, source_folder, trainingset_file_name):
        self.source_folder = source_folder
        self.trainingset_file_name = trainingset_file_name
        self.source_csv_path = source_folder + '/' + trainingset_file_name
        self.text_clf = None
        self.log_filename = source_folder + '/' + os.path.splitext(os.path.basename(self.trainingset_file_name))[0] + '_log_file.csv'
        self.predicted_filename = source_folder + '/' + os.path.splitext(os.path.basename(self.trainingset_file_name))[0] + '_preicted.csv'
        self.log_file_writer = open(self.log_filename, 'w', encoding='utf-8')
        self.multiLabelBinarizer = None

        return


    def train(self):
        self.WriteToLogFile('Start of multi-label review classification for different features!')
        data_frame = pd.read_csv(self.source_csv_path, index_col=None, header=0)

        data_frame.index.name = 'Sno'
        data_frame.insert(len(data_frame.columns), 'multi_label_index', 'list')

        labels_str = data_frame['multi_labels'][3]
        if ';' in labels_str:
            separator = ';'
        else:
            separator = ','

        for index, row in data_frame.iterrows():
            multi_label = row['multi_labels']
            data_frame.set_value(index, 'multi_label_index', multi_label.split(separator))

        reviews = data_frame.loc[:, ['r_no.', 'Comment']].as_matrix()

        target = data_frame.loc[:, 'multi_label_index'].tolist()

        docs_train_full, docs_test_full, y_train, y_test = train_test_split(reviews, target, test_size=0.25, random_state=None)

        docs_train = docs_train_full[:, 1]

        docs_test = docs_test_full[:, 1]

        docs_test_Rids = docs_test_full[:, 0]
        self.WriteToLogFile('Training set size = ' + str(len(docs_train)) + ', test set size = ' + str(len(docs_test)))
        self.multiLabelBinarizer = MultiLabelBinarizer()
        Y = self.multiLabelBinarizer.fit_transform(y_train)

        pipeline = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', OneVsRestClassifier(LogisticRegression()))])

        self.text_clf = pipeline.fit(docs_train, Y)

        predicted = self.text_clf.predict(docs_test)

        predicted_y_labels = self.multiLabelBinarizer.inverse_transform(predicted)

        perf_measures = ComputePerformanceMeasuresForMultiLabelClassification(y_test, predicted_y_labels)

        self.WriteToLogFile(perf_measures)

        data_frame.insert(len(data_frame.columns), 'multi_label_predicted', '')

        Y_pre = self.multiLabelBinarizer.transform(predicted_y_labels)
        Y_act = self.multiLabelBinarizer.transform(y_test)

        for item, predicted_labels, act_labels in zip(docs_test_Rids, predicted_y_labels, y_test):
            data_frame.set_value(item, 'multi_label_predicted', predicted_labels)

        print('Multi label measures : ' + str(ComputePerformanceMeasuresForMultiLabelClassification(y_test, predicted_y_labels)))

        data_frame.to_csv(self.predicted_filename, encoding='utf-8')

        self.WriteToLogFile('Results written to: ' + self.predicted_filename)

        self.WriteToLogFile('End of multi-label review classification for different features!')

        return self.text_clf


    def WriteToLogFile(self, message):
        self.log_file_writer.write(GetCurrentTimestamp() + ':= ' + message + '\n')
