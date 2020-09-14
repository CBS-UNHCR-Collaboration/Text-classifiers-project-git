"""
Raghava (2000211): This is the latest classifier file.
Use this one and do proper configuration
"""

import csv
import glob
import os
import sys
import string
import pathlib
import os.path
from os import path

from classification_scripts.classification_helper_functions import LoadStopwords

parentdir = os.path.dirname(__file__)
sys.path.insert(0, parentdir)
parenttoparent = os.path.dirname(parentdir)
sys.path.insert(0, parenttoparent)

import pandas as pd
from sklearn.model_selection import train_test_split

from ClassifierUtilities import *
from Classifiers.LinearSVCClassifier import LinearSVCClassifier
from Classifiers.LogisticRegressionClassifier import LogisticRegressionClassifier
from Classifiers.NaiveBayesMultinomialClassifier import NaiveBayesMultinomialClassifier
from Classifiers.PassiveAggressiveClassifierCustom import PassiveAggressiveClassifierCustom
from Classifiers.SVMSGDClassifier import SVMSGDClassifier


"""
The program assumes the following folder structure under the root.

root_folder
    -training
    -stopwords
    -data
    -results 

"""


# Change 1: changes the source folder path
#source_folder = '/Users/raghava/Dropbox (CBS)/cbs-research/Solutions/python-sols-new/' \
#                'twitter-premium-api-project-Fakenews/machine-learning/text-classification'

# source_folder = 'C:/Users/rrm.itm/Dropbox/cbs-research/research~projects/Researchers-CBS_INO-Trine/text' \
#                 '-classification/version-2.0 '

root_folder = '/Users/raghava/Dropbox (CBS)/cbs-research/Solutions/python-sols-new/' \
              'twitter-premium-api-project-Fakenews/machine-learning/' \
              'text-classification-v2.0/5G-Wuhan-Gates-combined/RIAS-Coding-Combined'

# changes 2: change the folder path
training_set_file = 'RIAS-training-data-combined.xlsx'

source_Excel_path = root_folder + '/training/' + training_set_file

stop_words_file_path = root_folder + '/stopwords/' + 'custom_stopwords_all.txt'

excel_sheet_name = 'RIAS-Coding-Balanced'

text_column_name = 'Tweet_text'

label_column_name = 'RIAS-Coding'

sno_column_name = 'S.No'

test_data_set_ratio = 0.30

# classifier Model Name
#model_name = excel_sheet_name
model_name = label_column_name

# saving of predictions for test data.
test_excel_file_name = root_folder + '/training/' + excel_sheet_name + '_predicted.xlsx'

# Parameters for ensembler classifiers.

# Control bit to proceed with Classification.
proceed_to_classify = True

data_folder = root_folder + '/data/'

results_root_folder = root_folder + '/results/'

results_folder = results_root_folder + excel_sheet_name + '/'

# Log file configuration
log_file = results_folder + excel_sheet_name + '_performance_logfile.csv'

# Error file.
error_records_file = results_folder + excel_sheet_name + '_error_records_logfile.csv'

delimiter_data_file = ';'

delimiter_results_file = ';'

text_column_index_data_file = 4


# Check whether folder exists
pathlib.Path(results_folder).mkdir(parents=True, exist_ok=True)

# open the log writer
log_file_writer = open(log_file, 'w', encoding='utf-8')


# ********************** Part-00 variable and load   ****************************

dict1 = dict.fromkeys(string.punctuation)

dict2 = dict.fromkeys(string.digits)

punctuation_digits_filter = str.maketrans({**dict1, **dict2})


custom_stop_word_set = LoadStopwords(stop_words_file_path)

print('stop_word_list: ', custom_stop_word_set)


# ********************** Part-01 functions ****************************

def WriteToLogFile(message):
    log_file_writer.write(GetCurrentTimestamp() + ':= ' + message + '\n')


def PredictUsingClassifier(cls, test_docs, act_labels):
    WriteToLogFile('\n\n***** ' + cls.classifier_name + ' - start of processing *****')

    # train the classifier with inputs
    cls.train()

    predicted_labels = cls.predict(test_docs)

    gs_cls = cls.grid_train()

    predicted_labels = cls.grid_predict(test_docs)

    WriteToLogFile(cls.classifier_name + ' Grid Accuracy : ' + str(np.mean(predicted_labels == act_labels)))

    measures = cls.grid_preformance_measures(act_labels)

    WriteToLogFile(cls.classifier_name + '\n' + measures)

    output_feature_info = cls.get_most_informative_features()

    WriteToLogFile(cls.classifier_name + '\n' + output_feature_info)

    WriteToLogFile('***** ' + cls.classifier_name + ' - end of processing ***** \n')

    return predicted_labels


def PredictLabelUsingVoting(clsList, text_to_predict):
    predicted = -1

    predicted_values = []

    for cls in clsList:
        predicted_values.append(cls.predict([text_to_predict])[0])

    predicted_counts = np.bincount(predicted_values)

    predicted = np.argmax(predicted_counts)

    return predicted


def PreProcessText(text):
    """
    This function will convert the text into lower case and then remove numbers and punctuation
    and optionally remove the supplied stopwords.

    :param text:
    :return:
    """

    # replace punctuation and digits
    processed_text = text.lower().translate(punctuation_digits_filter)
    return ' '.join([w for w in processed_text.strip().split(' ') if w not in custom_stop_word_set])



# ****************************** 2. Extracting labels **********************
WriteToLogFile('Start of classification with mutiple classifiers!')

data_frame_full = pd.read_excel(source_Excel_path, sheet_name=excel_sheet_name, index_col=None)

print('data_frame_full.shape: ', data_frame_full.shape)


# validating text value column and removing null values.
data_frame = data_frame_full.loc[data_frame_full[text_column_name].notnull(), :]

data_frame = data_frame.loc[data_frame[label_column_name].notnull(), :]
# validating the label column..

# adding sno_column_name to the dataset.
data_frame.insert(loc=0, column=sno_column_name, value=np.arange(len(data_frame)))

print('data_frame.shape: ', data_frame.shape)

data_frame.index.name = 'index_col'

# just to make sure that label column is string type
data_frame[label_column_name] = data_frame[label_column_name].astype(str)

# convert the case and strip sapces in labels
data_frame[label_column_name] = data_frame[label_column_name].str.strip()

data_frame[label_column_name] = data_frame[label_column_name].str.lower()

labels_factor = pd.factorize(data_frame[label_column_name])

target = labels_factor[0]

categories = labels_factor[1].base

print(categories)

#print(target)

categories = [str(cat) for cat in categories]

WriteToLogFile(','.join(categories))

data_frame[text_column_name] = data_frame[text_column_name].values.astype('U')

# to test the text values..
# counter = 1
# for record in data_frame[text_column_name]:
#     if len(record) < 100:
#         print(counter, ':', record)
#         counter += 1

for index, row in data_frame.iterrows():
    data_frame.loc[index, text_column_name] = PreProcessText(row[text_column_name])

print(data_frame[text_column_name][1:5])

# sys.exit(0)


text_docs_matrix = data_frame.loc[:, [sno_column_name, text_column_name]].as_matrix()

docs_train_full, docs_test_full, y_train, y_test = train_test_split(text_docs_matrix, target,
                                                                    test_size=test_data_set_ratio, random_state=None)

docs_train = docs_train_full[:, 1]

docs_test = docs_test_full[:, 1]

docs_test_Rids = docs_test_full[:, 0]

WriteToLogFile('Training set size = ' + str(len(docs_train)) + ', test set size = ' + str(len(docs_test)))

# ************************************************************************

# ****************************** 3. Traing Classifiers  **********************


# 1. NB classifier
nb_clr = NaiveBayesMultinomialClassifier(docs_train, y_train, categories, False)

nb_clr_predicted_y_test = PredictUsingClassifier(nb_clr, docs_test, y_test)

# 2. LinearSVCClassifier
svc_clr = LinearSVCClassifier(docs_train, y_train, categories, False)

svc_clr_predicted_y_test = PredictUsingClassifier(svc_clr, docs_test, y_test)

# 3. LogisticRegressionClassifier
lr_clr = LogisticRegressionClassifier(docs_train, y_train, categories, False)

lr_clr_predicted_y_test = PredictUsingClassifier(lr_clr, docs_test, y_test)

# 4. PassiveAggressiveClassifierCustom
pa_clr = PassiveAggressiveClassifierCustom(docs_train, y_train, categories, False)

pa_clr_predicted_y_test = PredictUsingClassifier(pa_clr, docs_test, y_test)

# 5. SVMSGDClassifier
sgd_clr = SVMSGDClassifier(docs_train, y_train, categories, False)

sgd_clr_predicted_y_test = PredictUsingClassifier(sgd_clr, docs_test, y_test)

print(str(len(nb_clr_predicted_y_test)) + ' : ' + str(len(svc_clr_predicted_y_test)))

print(str(len(lr_clr_predicted_y_test)) + ' : ' + str(len(pa_clr_predicted_y_test)))

# ****************************************************************************************


#  ****************************** 4. Print classification results  ****************************


voted_predictions_y_test = []

zipped_predictions = zip(docs_test_Rids, y_test, nb_clr_predicted_y_test, svc_clr_predicted_y_test,
                         lr_clr_predicted_y_test, pa_clr_predicted_y_test, sgd_clr_predicted_y_test)

# create a subset of main dataframe - test dataframe to writer the predicted values.

docs_test_index = [data_frame[data_frame[sno_column_name] == r_id].index[0] for r_id in docs_test_Rids]

data_frame_test = data_frame.loc[docs_test_index]

cls_list = [nb_clr, svc_clr, lr_clr, pa_clr, sgd_clr]

for cls in cls_list:
    data_frame_test.insert(len(data_frame_test.columns), cls.classifier_name, '')

# assign the value for voted classifier.
data_frame_test.insert(len(data_frame_test.columns), 'voted_label', '')

data_frame_test.insert(len(data_frame_test.columns), 'actual_label', '')

for (rid, y_act, nb, svc, lr, pa, sgd) in zipped_predictions:
    counts = np.bincount([nb, svc, lr, pa, sgd])
    voted_value = np.argmax(counts)
    voted_predictions_y_test.append(voted_value)

    r_index = data_frame_test[data_frame_test[sno_column_name] == rid].index[0]
    # assign values to respective columns in datafarme
    data_frame_test.set_value(r_index, nb_clr.classifier_name, categories[nb])
    data_frame_test.set_value(r_index, svc_clr.classifier_name, categories[svc])
    data_frame_test.set_value(r_index, lr_clr.classifier_name, categories[lr])
    data_frame_test.set_value(r_index, pa_clr.classifier_name, categories[pa])
    data_frame_test.set_value(r_index, sgd_clr.classifier_name, categories[sgd])
    data_frame_test.set_value(r_index, 'voted_label', categories[voted_value])
    data_frame_test.set_value(r_index, 'actual_label', categories[y_act])

WriteToLogFile('***** ' + ' Accuracy measures of voted classifier vs all classifiers  ***** \n')

WriteToLogFile('voted accuracy: ' + str(np.mean(voted_predictions_y_test == y_test)))

WriteToLogFile('NB accuracy: ' + str(np.mean(y_test == nb_clr_predicted_y_test)) + ' and NB vs Voted accuracy: ' + str(
    np.mean(voted_predictions_y_test == nb_clr_predicted_y_test)))

WriteToLogFile(
    'svc accuracy: ' + str(np.mean(y_test == svc_clr_predicted_y_test)) + ' and svc vs Voted accuracy: ' + str(
        np.mean(voted_predictions_y_test == svc_clr_predicted_y_test)))

WriteToLogFile('LR accuracy: ' + str(np.mean(y_test == lr_clr_predicted_y_test)) + ' LR and vs Voted accuracy: ' + str(
    np.mean(voted_predictions_y_test == lr_clr_predicted_y_test)))

WriteToLogFile('PA accuracy: ' + str(np.mean(y_test == pa_clr_predicted_y_test)) + ' PA and vs Voted accuracy: ' + str(
    np.mean(voted_predictions_y_test == pa_clr_predicted_y_test)))

WriteToLogFile(
    'SGD accuracy: ' + str(np.mean(y_test == sgd_clr_predicted_y_test)) + ' SGD and vs Voted accuracy: ' + str(
        np.mean(voted_predictions_y_test == sgd_clr_predicted_y_test)))

print('voted accuracy: ' + str(np.mean(voted_predictions_y_test == y_test)))


data_frame_test.to_excel(test_excel_file_name, encoding='utf-8')

# ******************************************************************************************************
if not proceed_to_classify:
    print(' exiting without classification!')
    sys.exit(0)


if input('Do you want to proceed with this accuracy?') != 'yes':
    print('exit without classification!')
    exit(0)


#  ****************************** 5. classification of new data using trained classifiers  ****************************


# create a results directory if it does not exists
pathlib.Path(results_folder).mkdir(parents=True, exist_ok=True)

for text_coprus_file in glob.iglob(data_folder + '*.csv'):
    results_file_path = results_folder + GetFilenameWithoutExtension(text_coprus_file) + '_classified.csv'
    with open(text_coprus_file, encoding='utf-8') as textfile, \
            open(results_file_path, 'w', newline='') as results_file, \
            open(error_records_file, 'w', newline='') as error_file:
        text_reader = csv.reader(textfile, delimiter= delimiter_data_file)
        results_writer = csv.writer(results_file, delimiter= delimiter_results_file, quoting=csv.QUOTE_MINIMAL)
        error_writer = csv.writer(error_file, delimiter=delimiter_results_file, quoting=csv.QUOTE_MINIMAL)
        header_row = next(text_reader)
        results_writer.writerow(header_row + ['model_name', 'predicted_voted_label'])
        error_writer.writerow(header_row)
        record_count = 0
        for row in text_reader:
            try:
                text_value = row[text_column_index_data_file]
                text_value = PreProcessText(text_value)
                if record_count == 2:
                    print('text at index 3: ', text_value)
                    if input('Check the sample text value to be classified. If you are statisfied then type "yes" ') \
                            != 'yes':
                        print('exit without classification!')
                        exit(0)

                predicted_val = PredictLabelUsingVoting(cls_list, text_value)
                predicted_label = categories[predicted_val]
                results_writer.writerow(row + [model_name, predicted_label])
                record_count = record_count + 1
                if record_count % 1000 == 0:
                    print(GetCurrentTimestamp() + ': completed_records:' + str(record_count))
            except:
                print('Error in classification! the error data record: ', row)
                error_writer.writerow(row)

    print(GetCurrentTimestamp() + ': completed the file:' + GetFilenameWithoutExtension(text_coprus_file))

WriteToLogFile('End of review classification with multiple classifiers for rating!')

log_file_writer.close()

# sample_array = ["I found it really hard to come to terms with the loss of my breast and getting used to my new appearance and this week have to come to terms again with the loss of my second breast as I have prevention surgery but as above im also still here fighting",
#                 "OMG I was on Tamoxifen for 5 years from 2001 to 2005 following grade 1 breast cancer.  I had horrific hot flushes almost continuously the whole time it was ghastly also gained nearly 2 stone in weight and my thyroid went a bit haywire.  I hated every minute but persevered as thought it was right thing to do.  I was also quite depressed at time I never thought of stopping as was worried about consequences"]
#
# count = 0
# for text in sample_array:
#     predicted_val=PredictLabelUsingVoting(cls_list,text)
#     count = count + 1
#     print('predict value_' + str(count) + categories[predicted_val])
#


# ******************************************************************************************************


print('done!')
