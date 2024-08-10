import pickle
import json
import numpy as np

# open the .pkl file
with open('../proof_graph.pkl', 'rb') as file:
    raw_data = pickle.load(file)

#the following class should allow for encoding of np objects in .json format
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
# write the .json file; 
# this .json file is used in creation of Dataset class
# Dataset class makes its own .json file to replace statements with vector embeddings
with open('../../Model/data/raw/data.json', 'w') as f:
    json.dump(raw_data, f, cls=NumpyEncoder)

# open the .json file; could also use pd.read_json
# with open("./data/data.json") as file:
    # json_load = json.load(file)