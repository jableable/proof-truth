# given a logical statement like the following from 36622 lcfrvalsnN:
# 400: '$p |- ( ( G e. R /\ x e. ( ._|_ ` ( L ` G ) ) ) -> E. f e. R x e. ( ._|_ ` ( L ` f ) ) )'
# use vocab.txt to find its embedding vector

# vector is of length equal to longest logical statement present in tag_proof.csv
# statements which do not fill vector are trailed by 0s

#one thing to fix: \ shows up as \\

import numpy as np

vector_size = 500   # 500 is placeholder for longest logical statement from tag_proof.csv
embedding = np.zeros(vector_size, dtype=int)   # initialize embedding vector
stmt = '( ( G e. R /\ x e. ( ._|_ ` ( L ` G ) ) ) -> E. f e. R x e. ( ._|_ ` ( L ` f ) ) )'

split_stmt = stmt.split()   # remove spaces and place each character into ordered list

with open("./Assets/vocab.txt", "r") as input:
    voc_index = {}
    for num, line in enumerate(input):  # correspond vocabulary with numbers in range 1-1598
        line = line.split() # get rid of \n
        voc_index[line[0]] = num+1  # shift by one so that 0 corresponds to space
    for num, char in enumerate(split_stmt):
        embedding[num] = voc_index[char]

print(embedding)


