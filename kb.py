# %%
import rdflib
import pickle

# %%
# insert a new feature and its rules into the Graph

def update_kb(feature : str, data : list, g : rdflib.Graph) -> rdflib.Graph:
    g.update(
        """
            INSERT DATA {
                ind:""" + feature + """ a classes:features
            }
       """
    )
    _s = '( ' + ' '.join(f'"{feature[:2]}:{ el }"' for el in data) + ' )'
    g.update(
        """
            PREFIX """ + feature[:2] + """:<URN:background:>
            INSERT DATA {
                ind:""" + feature + """ owl:oneOf """+ _s + """
            }
       """
    )
    return g
    
# %%
g = rdflib.Graph()
result = g.parse(file=open("rules/KB_NFT.n3"), format="n3")

# %%
# Load prepared classes for the features and prepare them for
# inserting into the kb
features = ['background', 'clothes', 'eyes', 'mouth']

for feature in features:
    with open(f"labels/{ feature }.p", 'rb') as fp:
        classes = pickle.load(fp)
    m = []
    for key in classes:
        for el in classes[key]:
            m.append(el)
    
    g = update_kb(feature, m, g)

# %%
g.serialize(destination='rules/KB_updated.n3')
# %%
