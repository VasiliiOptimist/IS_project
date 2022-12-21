# %%
import socket
import json
import pickle
import rdflib
import warnings

import pandas as pd

# %%
# Check that data in the KB.
def check_kb(data : dict):
    g = rdflib.Graph().parse(file=open("n3/KB_updated.n3"), format="n3")
    prepared_data = dict()
    _q = g.query(
            """
                SELECT DISTINCT ?value
                WHERE {
                    {
                        ?bnodes_first rdf:first ?value.
                    }
                }
            """
        )
    for key in data:
        kb_input_oneof = []
        for el in _q:
            if f"{ key[:2] }:" in el.asdict()['value']:
                kb_input_oneof.append(el.asdict()['value'][3:])
        if data[key] in kb_input_oneof:
            prepared_data[key] = data[key]
        else:
            print(f"{ data[key] } didn't exist in the knowledge database. It will be replaced by Empty")
            prepared_data[key] = 'Empty'
    return prepared_data


# check data in the knowledge database and create a df for the model
def prepare_data(data : dict) -> pd.DataFrame:
    _d = data.copy()
    _d.pop('price_eth', None)
    # check data in the knowledge database
    dict_after_kb = check_kb(_d)
    df = pd.DataFrame(columns=dict_after_kb.keys(), index=[1])
    df['eth (usd)'] = [data['price_eth']]
    df['seller_address'] = [200]
    for key in _d:
        with open(f"labels/{ key }.p", 'rb') as fp:
            classes = pickle.load(fp)
        for cl in classes:
            if dict_after_kb[key] in classes[cl]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df[key][1] = cl
    df = df[['seller_address','background','eyes','mouth','clothes','eth (usd)']]
    return df


# %%
def predict(data : dict) -> float:
    df_data = prepare_data(data)
    # download a trained model
    with open('models/rfr.p', 'rb') as fp:
        rfr = pickle.load(fp)

    return rfr.predict(df_data)



# %%
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('localhost', 50000))
print('The server is ready')
s.listen(1)
conn, addr = s.accept()
while 1:
    data = conn.recv(1024).decode('utf-8')
    if data:
        data_client = json.loads(data)
        print('The data from the client is recieved')

        print('Predicting price...')
        predicted_price = predict(*data_client)
        
        print('Send to the client...')
        conn.send(str(predicted_price).encode())
        print('Sent')
    else:
        break
    
conn.close()

# %%
