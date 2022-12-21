# %%
import pandas as pd
import requests
import numpy as np
import os

# %%
def convert_to_df(nft_data : dict) -> pd.DataFrame:
    df_data_nft = pd.DataFrame(nft_data.get('nft'), index=[nft_data.get('nft')['token_id']])
    df_data_price_datails = pd.DataFrame(nft_data.get('price_details'), index=[nft_data.get('nft')['token_id']])
    df_data = pd.merge(df_data_nft, df_data_price_datails, left_index=True, right_index=True)
    features = ['marketplace', 'transaction_date', 'seller_address']
    d_features = {}
    for feature in features:
        d_features[feature] = nft_data.get(feature)
    df_data = pd.merge(df_data, pd.DataFrame(d_features, index=[nft_data.get('nft')['token_id']]), left_index=True, right_index=True)
    return df_data


# a function adds new values from transaction data to the dataset
def get_df_from_transaction_data(df_nfts : pd.DataFrame, response : requests.models.Response) -> pd.DataFrame:
    for transaction in response.json()['transactions']:
        df_nfts = pd.concat([df_nfts, convert_to_df(transaction)], ignore_index=True)
    return df_nfts


# %%
# Download NFT's data  
# More information: https://docs.nftport.xyz/reference/retrieve-all-transactions

import requests

url = "https://api.nftport.xyz/v0/transactions"

querystring = {"chain":"ethereum","type":"sale","continuation": 'MTY2NDEzMzc3OV82MzQyY2FkNWRlNzEwMjUzM2M5ZWUyNzE='}

headers = {
    "Content-Type": "application/json",
    "Authorization": os.environ.get("KEY_API")
}

# get the first page
response = requests.request("GET", url, headers=headers, params=querystring)



# %%
transaction_features = ['contract_type', 'contract_address', 'token_id', 'asset_type', 'price', 'price_usd', 'marketplace', 'transaction_date']
df_nfts_transaction = get_df_from_transaction_data(pd.DataFrame(columns=transaction_features), response)

# %%
df_nfts_transaction.head()

# %%
import time
# Get information from the next pages
url = "https://api.nftport.xyz/v0/transactions"

# each page gives us 50 more nfts
num_pages = 5

for _ in range(num_pages):
    querystring = {
        "chain":"ethereum",
        "type":"sale",
        "continuation": response.json()['continuation']}

    headers = {
        "Content-Type": "application/json",
        "Authorization": os.environ.get("KEY_API"),
    }

    # get the first page
    # time.sleep(0.05)
    response = requests.request("GET", url, headers=headers, params=querystring)
    if response.status_code == 200:
        df_nfts_transaction = get_df_from_transaction_data(df_nfts_transaction, response)
    else:
        print(response.text)

# %%
response.json()['continuation']

# 'MTY2NDEzMzc3OV82MzQyY2FkNWRlNzEwMjUzM2M5ZWUyNzE='

# %%
df_nfts_transaction

# %% [markdown]
# 

# %%
# a funtction adds new values from metadata to the dataset
def add_nfts(df_nfts: pd.DataFrame, response: requests.models.Response) -> pd.DataFrame:
    nft = response.json().get('nft')
    if nft == None:
        return df_nfts
    row = np.array([nft.get(el) for el in df_nfts.keys()])
    df_nfts = pd.concat([df_nfts, pd.DataFrame([row], columns = df_nfts.keys())], ignore_index=True)
    return df_nfts


def get_attributes(sr_attributes: pd.Series, response: requests.models.Response) -> pd.Series:
    nft = response.json().get('nft')
    if nft == None or nft.get('metadata') == None:
        return sr_attributes
    sr_attributes = pd.concat(
        [sr_attributes, 
        pd.Series(  
            [nft.get('metadata').get('attributes')], 
            name='attributes',
            index=[nft.get('token_id')])
        ])
    return sr_attributes

# %%
import time
features = ['token_id', 'chain', 'contract_address']
df_nfts_contract = pd.DataFrame(columns = features)
sr_attributes = pd.Series(dtype='object', name='attributes')
for irow in df_nfts_transaction.index:
    contract = df_nfts_transaction['contract_address'][irow]
    token = df_nfts_transaction['token_id'][irow]
    url = f"https://api.nftport.xyz/v0/nfts/{contract}/{token}"
    time.sleep(0.05) # we have to wait some time due to the limitation of free access
    querystring = {"chain":"ethereum"}

    headers = {
        "Content-Type": "application/json",
        "Authorization": os.environ.get("KEY_API")
    }

    nft_response = requests.request("GET", url, headers=headers, params=querystring)
    print(nft_response.status_code)
    if nft_response.status_code == 200:
        df_nfts_contract = add_nfts(df_nfts_contract, nft_response)
        sr_attributes = get_attributes(sr_attributes, nft_response)
    else:
        print(nft_response.text)


# %%


# %%
df_nfts_transaction_1 = df_nfts_transaction.drop(['contract_address_x', 'contract_address_y'], axis=1)
df_nfts_transaction_1.drop_duplicates(subset=['token_id'], inplace=True)

# %%
df_nfts_contract_1 = df_nfts_contract.drop(['contract_address'], axis=1)
df_nfts_contract_1.drop_duplicates(subset=['token_id'], inplace=True)


# %%
df_nfts_contract_with_attr = pd.merge(df_nfts_contract_1, sr_attributes, how='left', left_on='token_id', right_index=True)
df_nfts_contract_1.head(15)

# %%
# df_nfts_contract_with_attr = pd.merge(df_nfts_contract_1, sr_attributes, left_on=['token_id'], right_index=True)
df_final = pd.merge(df_nfts_contract_with_attr, df_nfts_transaction_1,  how='left', on='token_id')
df_final.drop_duplicates(subset=['token_id'], inplace=True)
df_final.drop(['contract_type', 'chain', 'token_id', 'contract_address', 'asset_type', 'marketplace'], axis=1, inplace=True)
df_final.rename(columns={'price' : 'price_eth'}, inplace=True)
df_final.dropna(subset=['attributes'], inplace=True)
df_final

# %%
def convert_attributes(traits):
    res = dict()
    dict_synonyms = {'clothing' : 'clothes', 
                    'eye' : 'eyes',
                    }
    try:
        for trait in traits:
            if trait['trait_type'].lower() in dict_synonyms.keys():
                res[dict_synonyms[trait['trait_type'].lower()]] = trait.get('value')
            elif trait['trait_type'].lower() == 'eye':
                res['eyes'] = trait.get('value')
            else:
                res[trait['trait_type'].lower()] = trait.get('value')
        return res
    except (TypeError, AttributeError, KeyError):
        return None


# %%
df_final_1 = df_final.copy()
df_final_1['attributes'] = df_final['attributes'].apply(convert_attributes)
df_final_1


# %%
dict_traits = dict()
for attr in df_final_1.attributes:
    if attr:
        try:
            for trait in attr:
                dict_traits[trait] = dict_traits.setdefault(trait, 0) + 1
        except AttributeError:
            pass
        except TypeError:
            # print(trait)
            pass
        
preparred_for_df = []
for key in dict_traits:
    preparred_for_df.append([key, dict_traits[key]])
pd.DataFrame(preparred_for_df, columns=['trait_type','quantity']).sort_values(by=['quantity'], ascending=False).head(20)

# %%
attributes = ['background', 'eyes', 'body', 'mouth', 'head', 'clothes', 'hair']
df_separated_attributes = df_final_1.copy()


for attr in attributes:
    df_separated_attributes[attr] = df_final_1['attributes'].apply(lambda x: x.get(attr).lower() if x and type(x.get(attr)) is str else None)
df_separated_attributes.drop(['attributes'], axis=1, inplace=True)
df_separated_attributes

# %%
df_separated_attributes.fillna(value=np.nan, inplace=True)

# %%
df_separated_attributes.dropna(subset=['background', 'eyes', 'body'])


# %%
df_separated_attributes.to_csv('datasets/Dataset_0.3.csv', index=False)

# %%



