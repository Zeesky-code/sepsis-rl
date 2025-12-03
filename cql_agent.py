import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# --- Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Networks ---
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class CQLAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, gamma=0.99, cql_alpha=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.cql_alpha = cql_alpha
        
        # Q-Network (Double Q-Learning)
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.target_q1 = QNetwork(state_dim, action_dim, hidden_dim).to(DEVICE)
        self.target_q2 = QNetwork(state_dim, action_dim, hidden_dim).to(DEVICE)
        
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        
        self.optimizer = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        
    def select_action(self, state):
        state = torch.FloatTensor(state).to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            q1 = self.q1(state)
            q2 = self.q2(state)
            q = torch.min(q1, q2)
            return q.argmax().item()
            
    def train(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(DEVICE).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE).unsqueeze(1)
        
        # --- Standard Q-Learning Loss (Bellman Error) ---
        with torch.no_grad():
            # Target Q-value: r + gamma * max(Q_target(s', a'))
            next_q1 = self.target_q1(next_states)
            next_q2 = self.target_q2(next_states)
            next_q = torch.min(next_q1, next_q2)
            max_next_q = next_q.max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
            
        current_q1 = self.q1(states).gather(1, actions)
        current_q2 = self.q2(states).gather(1, actions)
        
        bellman_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # --- CQL Regularization Loss ---
        # Penalize high Q-values for unseen actions (push them down)
        # Maximize Q-values for actions in the dataset (pull them up)
        
        # LogSumExp of Q-values (soft-maximum over all actions)
        cql1_loss = torch.logsumexp(self.q1(states), dim=1).mean() - current_q1.mean()
        cql2_loss = torch.logsumexp(self.q2(states), dim=1).mean() - current_q2.mean()
        
        cql_loss = (cql1_loss + cql2_loss) * self.cql_alpha
        
        # Total Loss
        total_loss = bellman_loss + cql_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), bellman_loss.item(), cql_loss.item()
        
    def update_target_network(self, tau=0.005):
        for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
