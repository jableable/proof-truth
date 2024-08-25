# iterate over raw data.json file based on file limit (first 10000 files)
# extract pf label at each step and convert to num
# make dict count of frequencies

# if a frequency is n or less, change label to be unk (n=5 is default)
# when labels are reorganized, highest label is "unknown" bin

import pandas as pd
from statement_embedding import get_thm_label_num


# file limit is number of proof graphs to consider from data.json
def make_label_hist(file_limit):
 
    # read json file and restrict num of graphs considered
    pf_data = pd.read_json("./data/raw/data.json")
    pf_data = pf_data.iloc[:, : file_limit]

    # initialize histogram            
    label_count = {}

    # make num to label translation dict

    # iterate over all statements to initialize dict label entries
    # making frequency dict
    for graph in range(len(pf_data.columns)):
        if graph % 1000 == 0:   # loading progress
            print("processing graph",graph)
        stmt_lbl = pf_data[graph].graph_features[1]
        #if stmt_lbl == "nan":
            #stmt_num = 828
        #else:
        stmt_num = get_thm_label_num(stmt_lbl)
        # if stmt key is not in dict, initialize it
        if stmt_num not in label_count:
            label_count[stmt_num] = 1
        # shouldn't need commented out line below
        #elif stmt_num in label_count:
            #label_count[stmt_num] += 1
        for step in pf_data[graph].x:
            num = get_thm_label_num(step[1]["label"])
            # if label key is not in dict, initialize it
            # if label key is in dict, increase its value by 1
            if num not in label_count:
                label_count[num] = 1
            else:
                label_count[num] += 1
    print(label_count)
    return pf_data, label_count

# replace
def remove_infrequent_labels(dict, df, n):

    # initialize dict of infrequently used labels
    infrequent_labels = {}

    for key in dict.keys():
        if dict[key] <= n:
            infrequent_labels[key]=0   

    # iterate over pf_data to replace infrequent_labels with unk 
    print("replacing infrequent labels with unk...")
    for graph in range(len(df.columns)):
        if graph % 1000 == 0:   # loading progress
            print("processing graph",graph)
        stmt_lbl = pf_data[graph].graph_features[1]
        stmt_num = get_thm_label_num(stmt_lbl)
        if stmt_num in infrequent_labels:
            pf_data[graph].graph_features[1] = "unk"
        for i, step in enumerate(df[graph].x):
            num = get_thm_label_num(step[1]["label"])
            if num in infrequent_labels:
                df[graph].x[i][1]["label"] = "unk"

    print("writing to .json...")    
    # write the .json file
    df.to_json('./data/raw/'+str(len(df.columns))+'_relabeled_data_at_least_5.json')


if __name__ == "__main__":
    
    file_limit=10000
    pf_data, pf_dict = make_label_hist(file_limit)
    remove_infrequent_labels(dict=pf_dict, df=pf_data, n=5)