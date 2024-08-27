# proof-truth
This project was completed for the Erdős Institute - Summer 2024 - Deep Learning Bootcamp.

# Extracting proof graphs from Metamath
todo 

# Generating the .pt Graph Dataset

There are several files that must be generated (in order) so that the ultimate graph .pt file can be produced and fed into the GIN model. This is due, in part, to the following user-defined parameters. The file_size parameter determines how many proofs are considered from the initial dataset of ~42000 proofs. By default, this parameter is 10000. The other parameter is the minimum number of occurences for which a label needs to occur to be represented in the dataset. Label occurrences which fall below this threshold are relabeled to "unk" (for unknown). By default, this parameter is 5.

1. Beginning with the initial dataset [data.json](https://github.com/jableable/proof-truth/blob/main/Model/USE%20GIN%20model/data/raw/data.json), relabel and reindex labels by running [make_label_hist.py](https://github.com/jableable/proof-truth/blob/main/Model/USE%20GIN%20model/make_label_hist.py). This produces "10000_relabeled_data_at_least_5.json".

2. Feed "10000_relabeled_data_at_least_5.json" into [make_use_embeddings.py](https://github.com/jableable/proof-truth/blob/main/Model/USE%20GIN%20model/make_use_embeddings.py). This file produces 512-component vector embeddings of the first 10000 proofs as produced by [Google's Universal Sentence Encoder](https://research.google/pubs/universal-sentence-encoder/). The embeddings are stored in 10000_USE_embs.parquet.

3. We generate the initial graph .pt file by using the <code>ProofDataset</code> class (found [here](https://github.com/jableable/proof-truth/blob/main/Model/USE%20GIN%20model/use_dataset.py])) as instatiated at the beginning of [use_gin_model_w_hidden_conc.ipynb](https://github.com/jableable/proof-truth/blob/main/Model/USE%20GIN%20model/use_gin_model_w_hidden_conc.ipynb) as <code>pf_data</code>. Here, the read_name is the relabeled .json dataset (created in Step 1) used by the class, which by default is <code>read_name="10000_relabeled_data_at_least_5.json"</code>. The write_name, which is the name of the produced graph .pt file, is by default <code>write_name="10000_relabeled_data_at_least_5_w_stmts.pt"</code>

4. At this point, the model found in [use_gin_model_w_hidden_conc.ipynb](https://github.com/jableable/proof-truth/blob/main/Model/USE%20GIN%20model/use_gin_model_w_hidden_conc.ipynb) is runnable. To clarify, one more alteration is necessary within that file to correctly predict the final/conclusion node's label. This alteration is stored in <code>hidden_conc_pf_data</code>, and it's accomplished via the <code>HiddenConcProofDataset</code> class (found in the same file as  [<code>ProofDataset</code>](https://github.com/jableable/proof-truth/blob/main/Model/USE%20GIN%20model/use_dataset.py])). The output of that file is <code>overwritten_labels.pt</code>, and this is what the model should be trained on.

# Training the GIN and Results






