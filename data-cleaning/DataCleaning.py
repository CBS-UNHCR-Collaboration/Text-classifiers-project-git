import pandas as pd
import csv

from pathlib import Path



file_path = '/Users/raghava/Dropbox (CBS)/cbs-research/research~projects/' \
            'UNHCR/sentiment-analysis-data/text-classification-v2/' \
            'source-files-to-classify/Afghan-data/Copy of Afghan refugees COVID.xlsx'

path = Path(file_path)

filename_without_prefix = path.stem

sheet_name = 'Sheet1'
#folder_path = path.parent

text_column_name = 'Hit Sentence'


df = pd.read_excel(file_path, sheet_name)

print(df[text_column_name].head(10))

df = df.replace('\n', ' ', regex=True)

df = df.replace('\r', ' ', regex=True)

df = df.replace(';', ' ', regex=True)

df = df.replace(',', ' ', regex=True)

print('df.shape: ', df.shape)

df = df.loc[df[text_column_name].notnull(), :]

print('df.shape after cleaning: ', df.shape)

print(df[text_column_name].head(10))

df = df.reset_index(drop=True)

df.index.name = 'S.no'

df.to_csv(str(path.parent) + '/' + filename_without_prefix + '_cleaned.csv', sep=';')

print('Done exporting excel to csv')



