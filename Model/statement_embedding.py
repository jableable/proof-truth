# given a logical statement like the following proof step from lcfrvalsnN 
# (38836 in MM, 36622 in tag_proof.csv, 36284 in proof_graph.pkl):
# 400: '$p |- ( ( G e. R /\\ x e. ( ._|_ ` ( L ` G ) ) ) -> E. f e. R x e. ( ._|_ ` ( L ` f ) ) )'
# use vocab.txt to find its embedding vector

# vector is of length equal to longest logical statement present in tag_proof.csv
# statements which do not fill vector are trailed by 0s

import numpy as np


# create portion of embedding from theorem label; returns num of label from labels.txt
def get_thm_label_num(lbl):
    with open("../Assets/labels.txt", "r") as lbl_nums:
        for num, line in enumerate(lbl_nums):
            line = line.split() # get rid of \n
            if lbl in line:
                return num+1    #shift by 1 to match lines nums from labels.txt
    print(f"didn't find {lbl}")
    return


# create portion of embedding from theorem statement; returns embedding vector
def get_thm_stmt_emb(stmt):
    vector_size = 11547   # 11547 obtained as length of longest logical statement from find_longest_statement.py
    emb = np.zeros(vector_size, dtype=int)   # initialize embedding vector

    split_stmt = stmt.split()   # remove spaces from stmt and place each character into ordered list

    with open("../Assets/vocab.txt", "r") as input:
        voc_index = {}
        for num, line in enumerate(input):  # correspond vocabulary with numbers in range 1-1598 within voc_index dict
            line = line.split() # get rid of \n
            voc_index[line[0]] = num+1  # shift by 1 so that 0 corresponds to space
        for num, char in enumerate(split_stmt): # place index for char from voc_index into embedding
            emb[num] = voc_index[char]
    return emb


# combine embeddings from get_thm_label_num and get_thm_stmt_emb
def create_emb(lbl,stmt):
    #prepend emb from get_thm_stmt_emb(stmt) with num from get_thm_label_num(lbl)
    total_emb = np.insert(get_thm_stmt_emb(stmt),0,get_thm_label_num(lbl))
    return total_emb


if __name__ == "__main__":  
    
    #temporary example input of stmt and lbl
    lbl = 'rspcev'  
    stmt = '( ( G e. R /\\ x e. ( ._|_ ` ( L ` G ) ) ) -> E. f e. R x e. ( ._|_ ` ( L ` f ) ) )'
    print(create_emb(lbl,stmt))
    
