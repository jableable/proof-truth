import os.path as osp
import torch
from torch_geometric.data import Dataset
import pandas as pd
from statement_embedding import create_emb
from torch_geometric.data import Data, Dataset



# Create ProofDataset class, which is called by modelA.ipynb
# When calling ProofDataset(root="data/",file_limit = 100) in modelA.py, first 100 processed .pt files are generated
# See https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
# for more info

class ProofDatasetWithLabels(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, file_limit=None):
        self.file_limit = file_limit    # make file limit able to be called later
        super().__init__(root, transform, pre_transform, pre_filter)

    # indicate data file in data/raw to be processed
    @property
    def raw_file_names(self):
        return "data.json"

    # determine files to be checked when assembling dataset
    @property
    def processed_file_names(self):        
        self.data = pd.read_json(self.raw_paths[0]) # read in .json file
        # in next two if statements, either implement file_limit, or don't
        if self.file_limit is None:
            index_set = len(self.data.columns)
        if self.file_limit is not None:
            index_set = self.file_limit
        return ['data_'+str(idx)+'.pt' for idx in range(index_set)]

    # download from url not currently supported
    def download(self):
        pass

    # create missing .pt files according to file_limit
    def process(self):
        self.data = pd.read_json(self.raw_paths[0])
        self.data = self.data.drop(["edge_attr"])
        if self.file_limit is None:
            index_set = len(self.data.columns)
        if self.file_limit is not None:
            index_set = self.file_limit
        for index in range(index_set):   # loop through graphs specified by file_limit

            x = torch.Tensor([])    # initialize feature tensor for entire graph
            y = torch.Tensor([])    # initialize label tensor for entire graph

            for i, _ in enumerate(self.data[index]["x"]):   # loop through each proof step
                lbl = self.data[index]["x"][i][1]["label"]
                stmt = self.data[index]["x"][i][1]["statement"]
                emb = create_emb(lbl,stmt)
                x_emb = torch.Tensor([emb[1:]])
                y_emb = torch.Tensor([emb[0]])

                x = torch.cat((x,x_emb),dim=0)
                y = torch.cat((y,y_emb),dim=0)

            edge_index = torch.Tensor(self.data[index]["edge_index"])

            data = Data(x=x, y=y, edge_index=edge_index)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{index}.pt'))


            #if self.pre_filter is not None and not self.pre_filter(data):
                #continue

            #if self.pre_transform is not None:
                #data = self.pre_transform(data)

            

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
    

if __name__ == "__main__":
    test_obj = ProofDatasetWithLabels(root="data/",file_limit=100)
    print(test_obj)