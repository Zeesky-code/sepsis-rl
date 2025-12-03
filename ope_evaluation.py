import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from cql_agent import CQLAgent
import os

# --- Configuration ---
DATA_PATH = 'data/sepsis_trajectories.csv'
MODEL_PATH = 'models/cql_agent.pth'
HIDDEN_DIM = 128
GAMMA = 0.99

def compute_wis(agent, df):
    """
    Computes Weighted Importance Sampling (WIS) estimate of the policy value.
    V_WIS = sum(rho_t * r_t) / sum(rho_t)
    where rho_t = pi_e(a|s) / pi_b(a|s) (Importance Ratio)
    """
    
    # 1. Estimate Behavior Policy pi_b(a|s) from data
    # We'll use a simple empirical estimate: P(a|s) approx Count(s,a) / Count(s)
    # But since states are continuous, we'll train a small classifier (Behavior Cloning)
    # to estimate pi_b.
    
    print("Estimating Behavior Policy (Behavior Cloning)...")
    states = torch.FloatTensor(df[['hr', 'bp', 'o2', 'glucose']].values)
    actions = torch.LongTensor(df['action'].values)
    
    # Normalize states
    mean = states.mean(0)
    std = states.std(0)
    states = (states - mean) / (std + 1e-5)
    
    behavior_model = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 4) # Logits for 4 actions
    )
    optimizer = torch.optim.Adam(behavior_model.parameters(), lr=1e-3)
    
    # Train BC
    for _ in range(500):
        logits = behavior_model(states)
        loss = F.cross_entropy(logits, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print("Behavior Policy Estimated.")
    
    # 2. Compute Importance Ratios
    print("Computing Importance Ratios...")
    
    # Group by episode
    episodes = df.groupby('patient_id')
    
    wis_numerator = 0
    wis_denominator = 0
    
    per_episode_values = []
    
    for _, episode in episodes:
        ep_states = torch.FloatTensor(episode[['hr', 'bp', 'o2', 'glucose']].values)
        ep_states = (ep_states - mean) / (std + 1e-5)
        ep_actions = torch.LongTensor(episode['action'].values)
        ep_rewards = episode['reward'].values
        
        # Get pi_e (Evaluation Policy) probs
        # Since CQL is Q-learning (deterministic argmax), we can use softmax/epsilon-greedy
        # or just check if argmax matches action.
        # For OPE, a stochastic policy is better. Let's use Softmax on Q-values.
        with torch.no_grad():
            q_values = agent.q1(ep_states) # Use Q1
            pi_e_probs = F.softmax(q_values, dim=1)
            
            # Get pi_b (Behavior Policy) probs
            b_logits = behavior_model(ep_states)
            pi_b_probs = F.softmax(b_logits, dim=1)
            
        # Compute trajectory weight (rho)
        rho = 1.0
        discounted_return = 0
        
        for t in range(len(episode)):
            a = ep_actions[t]
            prob_e = pi_e_probs[t, a].item()
            prob_b = pi_b_probs[t, a].item()
            
            # Clip importance weight to reduce variance
            ratio = prob_e / (prob_b + 1e-5)
            ratio = min(ratio, 10.0) 
            
            rho *= ratio
            discounted_return += (GAMMA ** t) * ep_rewards[t]
            
        wis_numerator += rho * discounted_return
        wis_denominator += rho
        
        per_episode_values.append(discounted_return)
        
    wis_value = wis_numerator / (wis_denominator + 1e-5)
    avg_behavior_value = np.mean(per_episode_values)
    
    return wis_value, avg_behavior_value

def main():
    print("Loading Data and Model...")
    df = pd.read_csv(DATA_PATH)
    
    state_dim = 4
    action_dim = 4
    agent = CQLAgent(state_dim, action_dim, hidden_dim=HIDDEN_DIM)
    agent.q1.load_state_dict(torch.load(MODEL_PATH))
    
    print("Running Off-Policy Evaluation (Weighted Importance Sampling)...")
    wis_val, behavior_val = compute_wis(agent, df)
    
    print("\n--- OPE Results ---")
    print(f"Clinician (Behavior) Value: {behavior_val:.4f}")
    print(f"AI Agent (WIS) Value:       {wis_val:.4f}")
    
    if wis_val > behavior_val:
        print("\n✅ The AI Policy is estimated to be BETTER than the Clinician.")
    else:
        print("\n❌ The AI Policy is estimated to be WORSE than the Clinician.")

if __name__ == "__main__":
    main()
