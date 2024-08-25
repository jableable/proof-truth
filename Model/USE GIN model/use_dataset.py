import torch
import pandas as pd
from statement_embedding import get_thm_label_num
from torch_geometric.data import Data, InMemoryDataset
import numpy as np
import json 


# Create ProofDataset class
# When calling ProofDataset(root="data/"), a single .pt file is created
# # See https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
# for more info


class ProofDataset(InMemoryDataset):
    def __init__(self, root, read_name, write_name, transform=None, pre_transform=None, pre_filter=None, file_limit=None):
        self.file_limit = file_limit    # make file_limit able to be called later
        self.read_name = read_name
        self.write_name = write_name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        return [self.read_name]


    @property
    def processed_file_names(self):
        return [self.write_name]


    def download(self):
        pass

    # used to relabel y values so that all labels occur at least once
    # more explicitly, remove labels that don't occur, and reindex remaining labels
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

        # initialize data lists to be saved as test.pt later
        data_list1 = []
        data_list2 = []

        # read in USE embeddings to use as x features
        stmt_emb_df = pd.DataFrame(pd.read_parquet("../10000_USE_embs.parquet"))
        stmt_emb_arr = np.array(stmt_emb_df) 
        row_count = 0   # index currently used row of USE_embs.csv
        for index, _ in enumerate(self.data):
            if index < self.file_limit:
                if index % 1000 == 0:   # loading progress
                    print("processing graph",index)
                x1 = torch.Tensor([])    # initialize feature tensor for proof graphs
                y1 = torch.Tensor([])    # initialize label tensor for proof graphs
                x2 = torch.Tensor([])    # initialize feature tensor for stmt graphs
                y2 = torch.Tensor([])    # initialize label tensor for stmt graphs
                edge_index2 = torch.Tensor([])         # initialize edge tensor for stmt graphs
                
                num_nodes = 0   # reset node count for produced stmt graph
                for i, _ in enumerate(self.data[index]["x"]):   # loop through each pf step

                    lbl1 = self.data[index]["x"][i][1]["label"]  # get pf step string label
                    lbl1_num = get_thm_label_num(lbl1)    # convert string label to num 
                    if lbl1 == "$e":  # make nodes for each $e hypothesis of stmt
                        x2_emb = stmt_emb_arr[row_count]    # get USE embedding
                        x2_emb = torch.tensor([x2_emb]).float()
                        y2_emb = torch.Tensor([lbl1_num])           
                        x2 = torch.cat((x2,x2_emb),dim=0)                
                        y2 = torch.cat((y2,y2_emb),dim=0)
                        num_nodes += 1
                    if self.data[index]["x"][i][1] == self.data[index]["x"][-1][1]: # make node for conclusion of stmt
                        lbl = self.data[index]["graph_features"][1] # use label from overall graph
                        if lbl == "nan":
                            lbl_num = 828
                        else:
                            lbl_num = get_thm_label_num(lbl)
                        x2_emb = stmt_emb_arr[row_count]  
                        x2_emb = torch.tensor([x2_emb]).float()
                        y2_emb = torch.Tensor([lbl_num]) 
                        x2 = torch.cat((x2,x2_emb),dim=0)                
                        y2 = torch.cat((y2,y2_emb),dim=0)
                        num_nodes += 1                   
                    y1_emb = torch.Tensor([lbl1_num])    # return to proof graph
                    x1_emb = stmt_emb_arr[row_count]     # get USE embedding
                    x1_emb = torch.tensor([x1_emb]).float()
                    row_count += 1  # get to next USE embedding
                    x1 = torch.cat((x1,x1_emb),dim=0)                
                    y1 = torch.cat((y1,y1_emb),dim=0)
                destination_vertex = [i for i in (range(num_nodes-1))]
                assumption_vertices = [num_nodes-1 for _ in range(num_nodes-1)]
                cur_edge_index = torch.Tensor([destination_vertex,assumption_vertices])

                # compile edge_indices, make data objects, and append them to data_list to be saved later
                edge_index1 = torch.Tensor(self.data[index]["edge_index"])
                edge_index2 = torch.cat((edge_index2,cur_edge_index), dim=0)

                pf_data = Data(x=x1, y=y1, edge_index=edge_index1)
                stmt_data = Data(x=x2, y=y2, edge_index=edge_index2)

                data_list2.append(stmt_data) 
                data_list1.append(pf_data)

        # combine stmt and proof tree graphs
        data_list = data_list2+data_list1

        # now we fix labels by removing unused ones and shifting to fill in the gaps
        # create y tensor to hold all y labels; used to make label translation dict
        old_labels = torch.Tensor([])
        for data in data_list:
            old_labels = torch.concat((old_labels,data.y),dim=0)
        class_corr = self.make_lbl_corr_dict(old_labels)
        self.class_corr = class_corr

        with open(self.write_name+"class_corr","w") as output:
            output.write(json.dumps(self.class_corr))

        # overwrite old labels
        for data in data_list:
            data.y = torch.tensor([class_corr[x.item()] for x in data.y.to(int)])
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]


        torch.save(self.collate(data_list), self.processed_paths[0])



# the following class is used to relabel an instantiation of ProofDataset
# this relabeling is performed in use_gin_model_w_hidden_conc.ipynb

class HiddenConcProofDataset(InMemoryDataset):
    def __init__(self, root, read_name, write_name, data_list=None, transform=None, pre_transform=None, pre_filter=None, file_limit=None):
        self.data_list = data_list
        self.read_name = read_name
        self.write_name = write_name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self):
        return [self.read_name]


    @property
    def processed_file_names(self):
        return [self.write_name]


    def download(self):
        # Download to `self.raw_dir`.
        #path = download_url(url, self.raw_dir)
        pass    

    def process(self):
        self.data = self.data_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(self.data_list), self.processed_paths[0])



if __name__ == "__main__":
    
    # make a file called test.pt in ./data/processed
    test_obj= ProofDataset(root="data/",file_limit=3)
    # print label dictionary
    print(test_obj.class_corr)