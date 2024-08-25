# given a logical statement like the following proof step from lcfrvalsnN 
# (38836 in MM, 36622 in tag_proof.csv, 36284 in proof_graph.pkl):
# 400: '$p |- ( ( G e. R /\\ x e. ( ._|_ ` ( L ` G ) ) ) -> E. f e. R x e. ( ._|_ ` ( L ` f ) ) )'
# use vocab.txt to find its embedding vector

# vector is of length equal to longest logical statement present in tag_proof.csv
# statements which do not fill vector are trailed by 0s

import numpy as np


# make label index for get_thm_label_num
def create_lab_index():
    with open("../../Assets/true_labels.txt", "r") as input:
        lab_index = {}
        for num, line in enumerate(input):  # correspond labels with numbers in range 1-45332 within label_index dict
            line = line.split() # get rid of \n
            lab_index[line[0]] = num  # shift by 1 to match .txt file
        return lab_index
    

# create portion of embedding from theorem label; returns num of label from labels.txt
def get_thm_label_num(lbl):
    lab_index = create_lab_index()
    if lbl == None:
        print("nan!")
        return lab_index["nan"]
    return lab_index[lbl]  

# inverse of get_thm_label_num; returns label
def num_to_label(num):
    lab_index = create_lab_index()
    inv_lab_index = {v: k for k, v in lab_index.items()}
    return inv_lab_index[num]



# make vocabulary index for get_thm_stmt_emb
def create_voc_index():
    with open("../../Assets/vocab.txt", "r") as input:
        voc_index = {}
        for num, line in enumerate(input):  # correspond vocabulary with numbers in range 1-1598 within voc_index dict
            line = line.split() # get rid of \n
            voc_index[line[0]] = num+1  # shift by 1 so that 0 corresponds to space
        return voc_index


# create portion of embedding from theorem statement; returns embedding vector
def get_thm_stmt_emb(stmt):
    vector_size = 11547   # 11547 obtained as length of longest logical statement from find_longest_statement.py
    emb = np.zeros(vector_size, dtype=int)   # initialize embedding vector

    split_stmt = stmt.split()   # remove spaces from stmt and place each character into ordered list
    voc_index = create_voc_index()  
    for num, char in enumerate(split_stmt): # place index for char from voc_index into embedding
        emb[num] = voc_index[char]
    return emb


# inverse of get_thm_stmt_emb; returns statement
def emb_to_stmt(embedding):
    new_stmt = ""
    voc_index = create_voc_index()
    inv_voc_index = {v: k for k, v in voc_index.items()}
    for num in embedding[1:]:   # skip over number from label
        if num != 0:
            new_stmt += inv_voc_index[num]+" "
    new_stmt = new_stmt[:len(new_stmt)-1]        # remove space at end
    return new_stmt


# combine embeddings from get_thm_label_num and get_thm_stmt_emb
def create_emb(lbl,stmt):
    #prepend emb from get_thm_stmt_emb(stmt) with num from get_thm_label_num(lbl)
    lbl_emb = get_thm_label_num(lbl)
    total_emb = np.insert(get_thm_stmt_emb(stmt),0,lbl_emb)
    return total_emb


if __name__ == "__main__":  
    
    #example input of stmt and lbl
    lbl = '$e'  
    stmt = '( ( G e. R /\\ x e. ( ._|_ ` ( L ` G ) ) ) -> E. f e. R x e. ( ._|_ ` ( L ` f ) ) )'
    print(create_emb(lbl,stmt))
    print(emb_to_stmt(create_emb(lbl,stmt)))
    
    print(create_lab_index()["idi"])
    print(get_thm_label_num("$e"))
    print(get_thm_label_num("impbii"))