import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os


class MyDataset(Dataset):
    def __init__(self, filename):
        f = open(filename + '.txt','r')
        self.indices = ast.literal_eval(f.readline())
        f.close()
        self.A = torch.load(filename + 'A.pt')
        self.B = torch.load(filename + 'B.pt')
        self.C = torch.load(filename + 'C.pt')
        self.D = torch.load(filename + 'D.pt')

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):

        i = self.indices[idx]
        return self.A[i].to(device), self.B[i].to(device), self.C[idx].to(device), self.D[idx].to(device)

class EnhancedCrossAttentionModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, num_hidden_layers=3):
        super(EnhancedCrossAttentionModel, self).__init__()
        

        self.query_layer = nn.Linear(input_dim, hidden_dim)
        self.key_layer = nn.Linear(input_dim, hidden_dim)
        self.value_layer = nn.Linear(input_dim, hidden_dim)
        self.c_projection = nn.Linear(input_dim, hidden_dim)


        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(nn.Dropout(0.3))  

        self.fc = nn.Linear(hidden_dim, 1) 

    def forward(self, A, B, C):
        # A: [n, 128]
        # B: [128]
        # C: [128]
        
        keys = self.key_layer(A) # [n, hidden_dim]
        queries = self.query_layer(B)  # [hidden_dim]
        values = self.value_layer(A) 


        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))  # [n, n]
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [n, n]

        context_vector = torch.matmul(attention_weights, values)  # [n, hidden_dim]
        context_vector = torch.sum(context_vector, dim=0)  # [hidden_dim]
        

        combined_vector = context_vector + self.c_projection(C)  # [hidden_dim]
        for layer in self.hidden_layers:
            combined_vector = layer(combined_vector)

        output = self.fc(combined_vector) 


        return output.squeeze(-1)

def train_model(model, dataloader, num_epochs=10, learning_rate=0):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}...")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0.0
        for A, B, C, D in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}",mininterval=11):

            optimizer.zero_grad()
            output = model(A, B, C)
            loss = criterion(output, D)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss.append(total_loss / len(dataloader))
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss}')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }, f"model_epoch_{epoch+1}.pth")
        evaluate_model(model, test_dataloader)


def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for A, B, C, D in dataloader:
            output = model(A, B, C)
            loss = criterion(output, D)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    test_loss.append(avg_loss)
    print(f'Test Loss:  {test_loss}')

    return avg_loss



batch_size = 32
hidden_dim = 1024
num_hidden_layers = 3

checkpoint_path = 'model_epoch_10.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device = {device}')


print('start data loading')
train_dataset = MyDataset('train_data_128')
test_dataset = MyDataset('test_data_128')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
print('start model init')
model = EnhancedCrossAttentionModel(input_dim=128, hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers).to(device)


train_loss = []
test_loss = []
print('start training')
train_model(model, train_dataloader, num_epochs=300, learning_rate=5e-4)

