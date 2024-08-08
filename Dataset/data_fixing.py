import re
from collections import defaultdict
import pandas as pd
from metamath import Metamath
import ast




df = pd.read_csv('tag_proof.csv',dtype={'tag': str})
tag_dict = df.set_index('tag').to_dict(orient='index')
for tag in tag_dict:
    nodes = tag_dict[tag]['node']
    if isinstance(nodes, str):
        nodes = ast.literal_eval(nodes)
    new_nodes = {}
    for node in nodes:
        if node.startswith("@"):
            continue
        new_nodes[node] = {}
        for k in nodes[node]:
            new_nodes[node][k] =  [int(j) for j in nodes[node][k].split(',')]
    tag_dict[tag]['node'] = new_nodes
        
df2 = pd.DataFrame.from_dict(tag_dict, orient='index')
df2.to_csv('tag_proof2.csv')
