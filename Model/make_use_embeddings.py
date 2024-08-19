import tensorflow_hub as hub
import pandas as pd
import numpy as np


# number of proof trees to make embeddings for
index_set = 5000

# make database from json file
stmts = pd.read_json("./data/raw/data.json")

# import USE model
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)
print("module %s loaded" % module_url)
def embed(input):
  return model(input)

emb_list = np.array([embed([stmts[index]["x"][i][1]["statement"]]) for index in range(index_set) for i, _ in enumerate(stmts[index]["x"])])
emb_list = emb_list.reshape(-1,512)
emb_df = pd.DataFrame(emb_list)
print("Beginning csv creation...")
emb_df.to_csv('USE_embs.csv', header=None, index=None)
print("Done!")