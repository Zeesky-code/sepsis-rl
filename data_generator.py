import numpy as np
import pandas as pd
import os

# --- Configuration ---
NUM_PATIENTS = 1000
MAX_STEPS = 20
OUTPUT_DIR = 'data'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def generate_sepsis_data():
    """
    Generates synthetic sepsis patient trajectories.
    State: [HR, BP, O2, Glucose]
    Action: [Vasopressors (0/1), Fluids (0/1)] (Discrete: 0=None, 1=Vaso, 2=Fluids, 3=Both)
    Reward: +100 for discharge (BP normalized), -100 for death (BP crash)
    """
    all_transitions = []
    
    for patient_id in range(NUM_PATIENTS):
        # Initial State: High HR, Low BP (Septic Shock)
        hr = np.random.normal(110, 10)
        bp = np.random.normal(60, 10)
        o2 = np.random.normal(90, 5)
        glucose = np.random.normal(150, 20)
        
        for step in range(MAX_STEPS):
            state = [hr, bp, o2, glucose]
            
            # Clinician Policy (Suboptimal but okay)
            # If BP is low, give Vasopressors (Action 1) or Fluids (Action 2)
            if bp < 65:
                action = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            elif hr > 100:
                action = np.random.choice([0, 2], p=[0.3, 0.7])
            else:
                action = 0 # Do nothing
                
            # Environment Dynamics (The "Physics")
            # Vasopressors increase BP, increase HR slightly
            # Fluids increase BP, lower HR
            if action == 1: # Vaso
                bp += np.random.normal(10, 2)
                hr += np.random.normal(2, 1)
            elif action == 2: # Fluids
                bp += np.random.normal(5, 2)
                hr -= np.random.normal(5, 1)
            elif action == 3: # Both
                bp += np.random.normal(15, 3)
                hr -= np.random.normal(2, 1)
            else: # None
                bp -= np.random.normal(2, 1) # Natural deterioration
                
            # Noise
            hr += np.random.normal(0, 2)
            bp += np.random.normal(0, 2)
            o2 += np.random.normal(0, 1)
            
            next_state = [hr, bp, o2, glucose]
            
            # Reward & Termination
            done = False
            reward = 0
            
            if bp > 120: # Stabilized -> Discharge
                reward = 1.0
                done = True
            elif bp < 40: # Crash -> Death
                reward = -1.0
                done = True
            elif step == MAX_STEPS - 1: # Time limit
                reward = 0
                done = True
            else:
                # Intermediate reward for keeping BP in safe range
                if 65 <= bp <= 100:
                    reward = 0.01
                else:
                    reward = -0.01
            
            all_transitions.append({
                'patient_id': patient_id,
                'step': step,
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })
            
            if done:
                break
                
            # Update state
            hr, bp, o2, glucose = next_state
            
    # Save to CSV
    # Flatten state arrays for CSV
    flat_data = []
    for t in all_transitions:
        row = {
            'patient_id': t['patient_id'],
            'step': t['step'],
            'hr': t['state'][0], 'bp': t['state'][1], 'o2': t['state'][2], 'glucose': t['state'][3],
            'action': t['action'],
            'reward': t['reward'],
            'next_hr': t['next_state'][0], 'next_bp': t['next_state'][1], 'next_o2': t['next_state'][2], 'next_glucose': t['next_state'][3],
            'done': t['done']
        }
        flat_data.append(row)
        
    df = pd.DataFrame(flat_data)
    df.to_csv(os.path.join(OUTPUT_DIR, 'sepsis_trajectories.csv'), index=False)
    print(f"Generated {len(df)} transitions from {NUM_PATIENTS} patients.")

if __name__ == "__main__":
    generate_sepsis_data()
