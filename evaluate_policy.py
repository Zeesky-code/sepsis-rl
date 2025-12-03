import pandas as pd
import numpy as np
import torch
from cql_agent import CQLAgent
import matplotlib.pyplot as plt
import os

# --- Configuration ---
DATA_PATH = 'data/sepsis_trajectories.csv'
MODEL_PATH = 'models/cql_agent.pth'
HIDDEN_DIM = 128
OUTPUT_DIR = 'results'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def evaluate_policy():
    print("Loading Data and Model...")
    df = pd.read_csv(DATA_PATH)
    
    # Normalize States (Same as training)
    state_cols = ['hr', 'bp', 'o2', 'glucose']
    stats = {}
    for col in state_cols:
        stats[col] = {'mean': df[col].mean(), 'std': df[col].std()}
        
    state_dim = 4
    action_dim = 4
    
    agent = CQLAgent(state_dim, action_dim, hidden_dim=HIDDEN_DIM)
    agent.q1.load_state_dict(torch.load(MODEL_PATH))
    agent.q2.load_state_dict(torch.load(MODEL_PATH)) # Load same weights for eval
    
    print("Evaluating Policy...")
    
    # We will simulate new episodes using the environment logic from data_generator
    # But for "Offline" eval, we typically look at Value Estimates or do OPE (Off-Policy Evaluation).
    # Since we have the simulator logic, let's run a "Virtual Trial".
    
    num_episodes = 100
    clinician_rewards = []
    ai_rewards = []
    
    # --- AI Trial ---
    for _ in range(num_episodes):
        # Initial State
        hr = np.random.normal(110, 10)
        bp = np.random.normal(60, 10)
        o2 = np.random.normal(90, 5)
        glucose = np.random.normal(150, 20)
        
        total_reward = 0
        for step in range(20):
            # Normalize state for agent
            norm_state = [
                (hr - stats['hr']['mean']) / (stats['hr']['std'] + 1e-5),
                (bp - stats['bp']['mean']) / (stats['bp']['std'] + 1e-5),
                (o2 - stats['o2']['mean']) / (stats['o2']['std'] + 1e-5),
                (glucose - stats['glucose']['mean']) / (stats['glucose']['std'] + 1e-5)
            ]
            
            action = agent.select_action(norm_state)
            
            # Environment Dynamics (Same as generator)
            if action == 1: bp += np.random.normal(10, 2); hr += np.random.normal(2, 1)
            elif action == 2: bp += np.random.normal(5, 2); hr -= np.random.normal(5, 1)
            elif action == 3: bp += np.random.normal(15, 3); hr -= np.random.normal(2, 1)
            else: bp -= np.random.normal(2, 1)
            
            hr += np.random.normal(0, 2)
            bp += np.random.normal(0, 2)
            o2 += np.random.normal(0, 1)
            
            reward = 0
            done = False
            if bp > 120: reward = 100; done = True
            elif bp < 40: reward = -100; done = True
            elif step == 19: reward = 0; done = True
            else:
                if 65 <= bp <= 100: reward = 1
                else: reward = -1
                
            total_reward += reward
            if done: break
            
        ai_rewards.append(total_reward)
        
    # --- Clinician Trial (Baseline) ---
    for _ in range(num_episodes):
        hr = np.random.normal(110, 10)
        bp = np.random.normal(60, 10)
        o2 = np.random.normal(90, 5)
        glucose = np.random.normal(150, 20)
        
        total_reward = 0
        for step in range(20):
            # Clinician Logic
            if bp < 65: action = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            elif hr > 100: action = np.random.choice([0, 2], p=[0.3, 0.7])
            else: action = 0
            
            if action == 1: bp += np.random.normal(10, 2); hr += np.random.normal(2, 1)
            elif action == 2: bp += np.random.normal(5, 2); hr -= np.random.normal(5, 1)
            elif action == 3: bp += np.random.normal(15, 3); hr -= np.random.normal(2, 1)
            else: bp -= np.random.normal(2, 1)
            
            hr += np.random.normal(0, 2)
            bp += np.random.normal(0, 2)
            o2 += np.random.normal(0, 1)
            
            reward = 0
            done = False
            if bp > 120: reward = 100; done = True
            elif bp < 40: reward = -100; done = True
            elif step == 19: reward = 0; done = True
            else:
                if 65 <= bp <= 100: reward = 1
                else: reward = -1
                
            total_reward += reward
            if done: break
            
        clinician_rewards.append(total_reward)
        
    print(f"AI Mean Reward: {np.mean(ai_rewards):.2f}")
    print(f"Clinician Mean Reward: {np.mean(clinician_rewards):.2f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.boxplot([clinician_rewards, ai_rewards], labels=['Clinician (Baseline)', 'AI Agent (CQL)'])
    plt.title('Sepsis Treatment Outcome Comparison')
    plt.ylabel('Total Reward (Survival/Stability)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(OUTPUT_DIR, 'policy_comparison.png'))
    print(f"Comparison plot saved to {OUTPUT_DIR}/policy_comparison.png")

if __name__ == "__main__":
    evaluate_policy()
