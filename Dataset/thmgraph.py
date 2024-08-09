import pandas as pd
import ast
import re
import numpy as np
import pickle




def get_label(node: dict, k: int):
    if len(node)==0:
        return None
    for key, v in node.items():
        if k in v.keys():
            return key       #return the label string
    return None             #return None, if the the node is not among the labeled nodes

def get_substitution_dict(proof): #creates a dictionary with keys the steps that using repeated steps and values the steps that being repeated
    pattern=r'^@(\d+)'
    subst_dict={}
    for k,v in proof.items():
        match=re.match(pattern, v)
        if match:
            subst_dict[k]=int(match.group(1))
    return subst_dict 

def preprocess_edge_index(subst_dict: dict, edge_index: np.ndarray):
    if edge_index.size==0:
        return edge_index
    def mapping_funct(x):
        if x in subst_dict.keys():
            return subst_dict[x]
        else:
            return x
    vectorized_mapping_function = np.vectorize(mapping_funct)
    edge_index_preprocessed=vectorized_mapping_function(edge_index)
    return edge_index_preprocessed


def get_node_features(proof: dict, node: dict, hpt: dict):

    pattern_e = r'^\$e' #to match 1st two characters of $e statement
    #pattern_2=r'^..' 
    pattern_2=r'\$[^\s]*'
    pattern_at='^@' #to match @ 
    pattern_pipe_dash = r'.*?\|- ' #used in removing everything preceding |- symbol

    x=[]
    for i, (k, v) in enumerate(proof.items()):
        num=k
        label=None
        if re.match(pattern_at, v):  #write that later - what to do with @
            pass
        else:
            if node:
                label=get_label(node, k)  #get label if the text label exists; set label to None otherwise
                if label in hpt:
                    label = "$e"
            if not label:  
                match = re.match(pattern_2, v)
                label=match.group(0)    
            statement=re.sub(pattern_pipe_dash, '', v, count=1)  #remove everything preceeding |-
            x.append([i, {'num':num, 'label':label, 'statement': statement}])

    return x

def get_edge_index(node: dict) -> np.ndarray:     #edge index in format [2, num_edges]
    edge_index= []
    if not node:
        pass
    else:
        for key in node:
            for k in node[key].keys():
                for i in node[key][k]:
                    edge_index.append([i,k])
    return np.array(edge_index, dtype=np.longlong).transpose()


def get_edge_index_renumbered(node_index: list, edge_index: np.ndarray):  #renumber edge index into COO format
    if edge_index.size==0:
        return edge_index
    new_node_index=range(len(node_index))
    mapping_dict={n:i for n,i in zip(node_index, new_node_index)}
    vectorized_mapping_function = np.vectorize(lambda x: mapping_dict[x])
    edge_index_renumbered=vectorized_mapping_function(edge_index)
    return edge_index_renumbered




if __name__ == "__main__":

    df = pd.read_csv('../Assets/tag_proof.csv', index_col='tag')
    tag_dict = df.to_dict(orient='index')
    graph_dict={}

    for n, key in enumerate(tag_dict.keys()):
        
        proof=ast.literal_eval(tag_dict[key]['proof'])
        node=ast.literal_eval(tag_dict[key]['node'])
        hpt = ast.literal_eval(tag_dict[key]['hpt'])
        x=get_node_features(proof, node, hpt)
        edge_index=get_edge_index(node)
        subst_dict=get_substitution_dict(proof)
        edge_index=preprocess_edge_index(subst_dict, edge_index)
        edge_attr=edge_index.transpose()
        edge_index=get_edge_index_renumbered(list(proof.keys()), edge_index )
        graph_dict[n]={'graph_features': [n, key], # order of the theorem in set.mm, label of the theorem
                'x':x, #features of the nodes [num_nodes, num_node_features]
                'edge_index':edge_index, #edge index in COO format [2, num_edges]
                'edge_attr': edge_attr ,# Edge feature matrix with shape [num_edges, num_edge_features] - this is the edge_index without renumbering
                }
    

    with open('proof_graph.pkl', 'wb') as file:
        pickle.dump(graph_dict, file)
