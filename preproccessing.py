# %% [markdown]
# ### Dataset

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# %%
df_source = pd.read_csv("Dataset_0.3.csv")
df_source

# %%
df2 = df_source.loc[df_source.isnull().sum(axis=1) <= 3]
df3 = df2.drop(['body', 'head', 'hair'], axis=1)
df3

# %%
df4 = df3.copy().fillna('Empty')

# %%
background_to_replace = ['blue', 'yellow', 'red', 'green', 'black', 'purple', 'grey', 'gold', 
                        'white', 'orange', 'cyan', 'pink', 'skyscraper', 'teal', 'night', 'lime',
                        'aquamarine', 'olive', 'tan']
replaced_feature = 'background'
print(len(df4[replaced_feature].unique()))
for color in background_to_replace:
    df4.loc[(df4[replaced_feature] != np.nan) & df4[replaced_feature].str.contains(color), replaced_feature] = color

df4.loc[(df4[replaced_feature] != np.nan) & df4[replaced_feature].str.contains('gray'), replaced_feature] = 'grey'
print(len(df4[replaced_feature].unique()))

# %%
clothes_to_replace = ['blue', 'yellow', 'red', 'green', 'black', 'purple', 'grey', 
                    'gold', 'white', 'orange', 'cyan', 'pink', 'skyscraper', 
                    'teal', 'night', 'lime', 'toga', 'tee', 'shirt', 'suit',
                    'jacket', 'biker', 'sleeveless', 'lumberjack', 'turtleneck',
                    'coat', 'robe', 'pelt']
replaced_feature = 'clothes'
print(len(df4[replaced_feature].unique()))
for color in clothes_to_replace:
    df4.loc[(df4[replaced_feature] != np.nan) & df4[replaced_feature].str.contains(color), replaced_feature] = color

df4.loc[(df4[replaced_feature] != np.nan) & df4[replaced_feature].str.contains('gray'), replaced_feature] = 'grey'
print(len(df4[replaced_feature].unique()))

# %%
mouth_to_replace = ['brown', 'blue', 'yellow', 'red', 'green', 'black', 
                'purple', 'grey', 'gold', 'white', 'orange', 'cyan', 'pink',
                'grin', 'dumbfounded', 'smile', 'tongue', 'toothy', 'mouth', 
                'smirk', 'phoneme', 'biting', 'cigarette', 'pipe', 'open',
                'happy', 'teeth']
replaced_feature = 'mouth'
print(len(df4[replaced_feature].unique()))
for color in mouth_to_replace:
    df4.loc[(df4[replaced_feature] != np.nan) & df4[replaced_feature].str.contains(color), replaced_feature] = color

df4.loc[(df4[replaced_feature] != np.nan) & df4[replaced_feature].str.contains('gray'), replaced_feature] = 'grey'
print(len(df4[replaced_feature].unique()))

# %%
eye_to_replace = ['bionic', 'brown', 'blue', 'yellow', 'red', 'green', 'black', 
                'purple', 'grey', 'gold', 'white', 'orange', 'cyan', 'pink', 
                'fire', 'beady', '3d', 'happy', 'spider', 'closed', 'sleepy',
                'angry', 'zombie', 'coins', 'heart', 'eyepatch', 'hypnotized', 
                'confuse', 'wide', 'laser', 'blindfold', 'hazel']
replaced_feature = 'eyes'
print(len(df4[replaced_feature].unique()))
for color in eye_to_replace:
    df4.loc[(df4[replaced_feature] != np.nan) & df4[replaced_feature].str.contains(color), replaced_feature] = color

df4.loc[(df4[replaced_feature] != np.nan) & df4[replaced_feature].str.contains('gray'), replaced_feature] = 'grey'
print(len(df4[replaced_feature].unique()))

# %%
def count_values(col_name, df):    
    d = dict()
    for el in df[col_name].unique():
        d[el] = df.loc[df4[col_name] == el][col_name].count()
    return dict(sorted(d.items(), key=lambda item: item[1])[::-1])

def create_new_values(counted_values):
    d = dict([ (k, []) for k in range(1, 20)])
    for el in counted_values:
        if 1 <= counted_values[el] < 20:
            d[counted_values[el]] += [el]
    zeros = []
    for key in d:
        if len(d[key]) == 0:
            zeros.append(key)
    for z in zeros:
        d.pop(z, None)
    return d

def add_to_dataset(new_values, replaced_feature, df):
    for key in new_values:
        df.loc[(df4[replaced_feature] != np.nan) & df4[replaced_feature].isin(new_values[key]), replaced_feature] = str(key)


# %%
features = ['background', 'eyes', 'mouth', 'clothes']

df5 = df4.copy()

for feature in features:
    count_feature = count_values(feature, df5)
    d_new_feature = create_new_values(count_feature)
    add_to_dataset(d_new_feature, feature, df5)

df5

# %%
from sklearn import preprocessing
import pickle

df_transformed = df5.copy()

features = ['background', 'eyes', 'mouth', 'clothes']

for feature in features:
    le = preprocessing.LabelEncoder()
    le.fit(df_transformed[feature])

    d = dict(zip([i for i in range(len(le.classes_))], le.classes_))
    
    # Save a dictionary label:values for the further usage in a KB
    count_feature = count_values(feature, df4)
    d_new_feature = create_new_values(count_feature)
    for key in d:
        try:
            if int(d[key]) in list(d_new_feature.keys()): 
                d[key] = d_new_feature[int(d[key])]
        except ValueError:
            d[key] = [d[key]]
            pass        
    with open('labels/' + feature + '.p', 'wb') as fp:
        pickle.dump(d, fp, protocol=pickle.HIGHEST_PROTOCOL)
    le.transform(df5[feature])
    # Saving completed
    
    df_transformed[feature] = le.transform(df5[feature])




# %%
df_transformed.to_csv('datasets/Dataset_v1.csv', index=False)