import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ast
import torch

def preload(filename):

    df = pd.read_csv(filename)

    A = []
    B = []
    C = []
    D = []
    t = []
    for index, row in df.iterrows():
        A.append(ast.literal_eval(row['condition']))
        B.append(ast.literal_eval(row['conclusion']))
        C += ast.literal_eval(row['statement'])
        D += ast.literal_eval(row['distance'])
        t += [index]*len(ast.literal_eval(row['distance']))
    D = [np.log(d + 1) for d in D]   
     
    A_tensor = torch.tensor(A, dtype=torch.float32)
    print(A_tensor.shape)
    fileA = filename[:-4]+'A.pt'
    torch.save(A_tensor,fileA)
    B_tensor = torch.tensor(B, dtype=torch.float32)
    print(B_tensor.shape)
    fileB = filename[:-4]+'B.pt'
    torch.save(B_tensor,fileB)
    C_tensor = torch.tensor(C, dtype=torch.float32)
    print(C_tensor.shape)
    fileC = filename[:-4]+'C.pt'
    torch.save(C_tensor,fileC)
    D_tensor = torch.tensor(D, dtype=torch.float32)
    print(D_tensor.shape)
    fileD = filename[:-4]+'D.pt'
    torch.save(D_tensor,fileD)

    f = open(filename[:-4]+'.txt','w')
    f.write(str(t))
    f.close()


preload('train_data_128.csv')

