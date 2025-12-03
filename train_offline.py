import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from cql_agent import CQLAgent
import os

# --- Configuration ---
DATA_PATH = 'data/sepsis_trajectories.csv'
BATCH_SIZE = 64
EPOCHS = 20
HIDDEN_DIM = 128
LR = 3e-4
CQL_ALPHA = 0.1

# --- Dataset ---
class SepsisDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        
        # Normalize States
        state_cols = ['hr', 'bp', 'o2', 'glucose']
        for col in state_cols:
            self.data[col] = (self.data[col] - self.data[col].mean()) / (self.data[col].std() + 1e-5)
            self.data[f'next_{col}'] = (self.data[f'next_{col}'] - self.data[f'next_{col}'].mean()) / (self.data[f'next_{col}'].std() + 1e-5)
            
        self.states = self.data[state_cols].values.astype(np.float32)
        self.actions = self.data['action'].values.astype(np.int64)
        self.rewards = self.data['reward'].values.astype(np.float32)
        self.next_states = self.data[[f'next_{c}' for c in state_cols]].values.astype(np.float32)
        self.dones = self.data['done'].values.astype(np.float32)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx])

def main():
    print("Loading Sepsis Data...")
    dataset = SepsisDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Loaded {len(dataset)} transitions.")
    
    state_dim = 4
    action_dim = 4 # None, Vaso, Fluids, Both
    
    agent = CQLAgent(state_dim, action_dim, hidden_dim=HIDDEN_DIM, lr=LR, cql_alpha=CQL_ALPHA)
    
    print("Starting Offline Training (CQL)...")
    for epoch in range(EPOCHS):
        total_loss = 0
        total_bellman = 0
        total_cql = 0
        
        for batch in dataloader:
            loss, bellman, cql = agent.train(batch)
            agent.update_target_network()
            
            total_loss += loss
            total_bellman += bellman
            total_cql += cql
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Bellman: {total_bellman/len(dataloader):.4f} | CQL: {total_cql/len(dataloader):.4f}")
        
    # Save Model
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(agent.q1.state_dict(), 'models/cql_agent.pth')
    print("Training Complete. Model saved to models/cql_agent.pth")

if __name__ == "__main__":
    main()
