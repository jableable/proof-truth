import os.path as osp
import torch
import pandas as pd
from statement_embedding import create_emb, get_thm_label_num, create_lab_index
from torch_geometric.data import Data, Dataset, InMemoryDataset
import numpy as np


# Create ProofDataset class
# When calling ProofDataset(root="data/"), a single .pt file is
# # See https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
# for more info


class ProofDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, file_limit=None):
        self.file_limit = file_limit    # make file_limit, vocab_size, label_size able to be called later
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        return ["data.json"]


    @property
    def processed_file_names(self):
        return ["test.pt"]


    def download(self):
        # Download to `self.raw_dir`.
        #path = download_url(url, self.raw_dir)
        pass


    # used to relabel y values so that all labels occur at least once
    # more explicitlu, remove labels that don't occur, and reindex remaining labels
    def make_lbl_corr_dict(self,old_classes):
        
        old_classes = old_classes.to(int)
        old_classes = old_classes.tolist()
        old_classes = set(old_classes)  # remove duplicates
        old_classes = sorted(list(old_classes)) # get in order

        new_classes=[i for i in range(len(old_classes))]
        class_corr = {i:j for i,j in zip(old_classes,new_classes)}  # make dictionary correspondence between old and new
        print("class_corr dict is of form (old_label, new_label):",class_corr)
        return class_corr
    

    def process(self):
        self.data = pd.read_json(self.raw_paths[0])
        self.data = self.data.drop(["edge_attr"])

        # initialize data list to be saved as test.pt later
        data_list = []

        # read in USE embeddings to use as x features
        stmt_emb_df = pd.DataFrame(pd.read_csv("./USE_embs.csv", header=None))
        stmt_emb_arr = np.array(stmt_emb_df) 
        row_count = 0   # index currently used row of USE_embs.csv
        for index, _ in enumerate(self.data):
            if index < self.file_limit:
                x = torch.Tensor([])    # initialize feature tensor for entire graph
                y = torch.Tensor([])    # initialize label tensor for entire graph
                
                for i, _ in enumerate(self.data[index]["x"]):   # loop through each pf step
                    lbl = self.data[index]["x"][i][1]["label"]  # get pf step string label 
                    lbl_num = get_thm_label_num(lbl)    # convert string label to num
                    y_emb = torch.Tensor([lbl_num])
                    x_emb = stmt_emb_arr[row_count]     # get USE embedding
                    x_emb = torch.tensor([x_emb]).float()
                    row_count += 1  # get to next USE embedding
                    x = torch.cat((x,x_emb),dim=0)                
                    y = torch.cat((y,y_emb),dim=0)

                edge_index = torch.Tensor(self.data[index]["edge_index"])
                data = Data(x=x, y=y, edge_index=edge_index)
                data_list.append(data)

        # create y tensor to hold all y labels; used to make label translation dict
        old_labels = torch.Tensor([])
        for data in data_list:
            old_labels = torch.concat((old_labels,data.y),dim=0)
        class_corr = self.make_lbl_corr_dict(old_labels)
        self.class_corr = class_corr

        # overwrite old labels
        for data in data_list:
            data.y = torch.tensor([class_corr[x.item()] for x in data.y.to(int)])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]


        torch.save(self.collate(data_list), self.processed_paths[0])

            #if self.pre_filter is not None and not self.pre_filter(data):
                #continue

            #if self.pre_transform is not None:
                #data = self.pre_transform(data)

if __name__ == "__main__":
    
    # make a file called test.pt in ./data/processed
    test_obj= ProofDataset(root="data/",file_limit=3)
    # print label dictionary
    print(test_obj.class_corr)