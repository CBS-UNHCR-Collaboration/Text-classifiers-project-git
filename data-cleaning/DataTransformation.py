import pandas as pd
import numpy as np

#file_path = '/Users/raghava/data-analytics Dropbox/Raghava Rao Mukkamala/cbs-research/research~projects/Researchers-CBS_INO-Trine/text-classification/training-set/18.12.2019 ML FIL.xlsx'

file_path = "C:/Users/rrm.itm/Dropbox/cbs-research/research~projects/Researchers-CBS_INO-Trine/text-classification" \
            "/original-data/24.02.2020 - Royal-Theatre-post.xlsx"

#folder_path = '/Users/raghava/data-analytics Dropbox/Raghava Rao ' /
#              'Mukkamala/cbs-research/research~projects/Researchers-CBS_INO-Trine/text-classification/training-set/ '

folder_path = 'C:/Users/rrm.itm/Dropbox/cbs-research/research~projects/Researchers-CBS_INO-Trine/text-classification' \
              '/version-2.0/training-set/'

post_df= pd.read_excel(file_path, 'posts')

post_df = post_df.reset_index(drop=True)

post_df.index.name = 'S.no'

print(post_df.index.name)

print(post_df.columns)

standard_columns = post_df.columns[0:16].tolist()

print(standard_columns)

model_columns = post_df.columns[16:].tolist()

print(model_columns)



print('shape of original data: ', post_df.shape)

col_list = []

with pd.ExcelWriter(folder_path + 'training-posts-only.xlsx') as writer:
    for model in model_columns:
        col_list = standard_columns + [model]
        post_df_new = post_df.loc[post_df[model].notnull(),col_list]
        sheet_name = model[0:7]
        post_df_new = post_df_new.reset_index(drop=True)
        post_df_new.index.name = 'S.no'
        post_df_new.to_excel(writer, sheet_name = sheet_name)
        print(model, ' : ', post_df_new.shape)


# finally write the data file with index..


#writer = pd.ExcelWriter(folder_path + 'posts-data-set.xlsx', engine='xlsxwriter',options={'strings_to_urls': False})

post_df.to_excel(folder_path + 'data-set-posts-only.xlsx')

print('Done!')

