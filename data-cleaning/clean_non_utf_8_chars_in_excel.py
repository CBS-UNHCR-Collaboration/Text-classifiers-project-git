import pandas as pd
import csv

from pathlib import Path



file_path = '/Users/raghava/Dropbox (CBS)/cbs-research/research~projects/' \
            'UNHCR/sentiment-analysis-data/text-classification-v2/training/Training-Data.xlsx'

path = Path(file_path)

filename_without_prefix = path.stem

sheet_name = 'Intensity'
#folder_path = path.parent

text_column_name = 'Tweet'

df = pd.read_excel(file_path, sheet_name)

print(df[text_column_name].head(10))

non_utf_index = 0

line_index = 0

for value in df[text_column_name].values:
    print('line index:', line_index)
    line_index += 1

    #value2 = bytes(value, 'utf-8').decode('utf-8', 'ignore')

    for char in value:
        try:
            if ord(char) > 255:
                print(non_utf_index, '): ', char)
                non_utf_index += 1

        except:
            print(line_index, ': error in conversion ')
            continue

print('total non_utf_8 chars: ', non_utf_index)




