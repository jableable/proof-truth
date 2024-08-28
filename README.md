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

### Setup

The GIN (graph isomorphism network) model is a powerful variant of GNNs (graph neural networks). This architecture was introduced in 2018 by Xu et al. in the paper “How Powerful are Graph Neural Networks?”, and it is implemented in Pytorch-Geometric as [<code>GINConv</code>](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINConv.html). The overall idea of this model is to predict the label of a given node by incorporating nearby node features (similar to how a CNN groups together nearby pixels in computer vision).

The purpose of our model is to predict missing proof steps in proof trees. In particular, we are interested in predicting the final step, aka the conclusion node's label. The label of a node corresponds to the logical step required to reach that node from previous nodes, and the labels are obtained from Metamath as described above. This means that we are dealing with a supervised multi-class classification problem.

For all of the following results, our model was trained from scratch on 10,000 proof graphs and 10,000 statement graphs, for a total of 20,000 graphs. In this setup, there are 3,557 different labels (logical steps) for our model to choose from when making predictions. This wide variety of choices can seem daunting, but our model did surprisingly well. 

### Training

After training for 1000 epochs with a learning rate of .001 and using optimizer RMSprop, the training accuracy becomes very high (>96%). However, the validation accuracy is much lower (~55%), partly due to overfitting. We implemented dropout to help combat this overfitting, but like many deep learning architectures, GNNs struggle with overfitting. GNNs also struggle with something called oversmoothing, which is the problem of nearby node features becoming too influential on a given node, to the point where the given node loses its identity in the process and the graph features become homogeneous. 

While a validation accuracy of 55% is not terrible in our context, we have a more representative notion of accuracy. Ultimately, we predict the final logical step in a proof so that it can be verified by a proof assistant (which always knows whether a proof is correct or incorrect). The opportunity cost of making a mistake in a proof assistant is minimal; just undo the mistake and try again. For this reason, we relax the notion of a "correct" prediction in the following way: in our model, a prediction is obtained by taking the argmax of a 3,557 component vector. Our more representative "Top 5 Accuracy" instead takes the *five* highest arguments. If the correct prediction is in these five guesses, then the prediction is correct. 

### Final GIN Results

When restricting to conclusion nodes, which is our true application, the validation accuracy is XX% and the "Top 5 Accuracy" is XX%. This result is quite remarkable given the wide variety of proofs in our dataset and the large number (3,557) of different logical steps that could be chosen.

# Future Work

While training our GIN on graphs to make predictions is both powerful and useful, it would be more user-friendly if it were integrated in an existing proof assistant like Metamath or LEAN.

We also considered several different predictive models within this project. It would be interesting to integrate them together in an ensemble format to see if our results are improved.





