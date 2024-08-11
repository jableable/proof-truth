import os.path as osp
import torch
from torch_geometric.data import Dataset
import pandas as pd
from statement_embedding import create_emb
from torch_geometric.data import Data, Dataset



# Create ProofDataset class, which is called by modelA.py
# When calling ProofDataset(root="data/") in modelA.py, all processed .pt files are generated
# See https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
# for more info

class ProofDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return "data.json"

    @property
    def processed_file_names(self):
        self.data = pd.read_json(self.raw_paths[0])
        return ['data_'+str(idx)+'.pt' for idx, _ in enumerate(self.data)]

    def download(self):
        # Download to `self.raw_dir`.
        #path = download_url(url, self.raw_dir)
        pass

    def process(self):
        self.data = pd.read_json(self.raw_paths[0])
        for index, _ in enumerate(self.data):

            x = torch.Tensor([])

            for i, step in enumerate(self.data[index]["x"]):
                lbl = self.data[index]["x"][i][1]["label"]
                stmt = self.data[index]["x"][i][1]["statement"]
                emb = torch.Tensor([create_emb(lbl,stmt)])
                x = torch.cat((x,emb),dim=0)

            edge_index = torch.Tensor(self.data[index]["edge_index"])

            data = Data(x=x, edge_index=edge_index)

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