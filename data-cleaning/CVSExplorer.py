import csv


def getValuesFromCSV(filepath, delimiter, column_number):
    """

    :param filepath:
    :param delimiter:
    :param column_number:
    :return: a list containing the first 5 elements from the column values
    """
    index = 0

    list_vals = []

    with open(filepath, encoding='utf-8') as textfile:
        text_reader = csv.reader(textfile, delimiter=delimiter)
        for row in text_reader:
            text_value = row[column_number]
            list_vals.append(text_value)
            index += 1
            if index == 5:
                return list_vals



filepath = '/Users/raghava/data-analytics Dropbox/Raghava Rao Mukkamala/' \
           'cbs-research/misc+archieve/new/Copy of COVID Afghan refugees v2_cleaned.csv'

print(getValuesFromCSV(filepath,';',7))