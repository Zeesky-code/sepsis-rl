# The AI Clinician: Offline Reinforcement Learning for Sepsis Treatment

**Day 3 of #AdventOfHealthcareML**

This project implements an **Offline Reinforcement Learning** agent using **Conservative Q-Learning (CQL)** to optimize sepsis treatment strategies (Vasopressors & IV Fluids) purely from historical patient data.

## The Problem
In healthcare, we cannot train RL agents by "exploring" on real patients (trial and error is dangerous). We must learn the optimal policy **offline**, using only static historical datasets (e.g., MIMIC-III).

## Methodology
1.  **Synthetic Environment**: A simulation of sepsis physiology (Heart Rate, BP, O2) and response to treatments.
2.  **Offline Dataset**: We generate a dataset of patient trajectories using a "Clinician Policy" (a rule-based baseline).
3.  **Conservative Q-Learning (CQL)**: We train a Deep Q-Network that penalizes Q-values for actions not seen in the dataset. This prevents the agent from overestimating the value of risky, unknown actions—a critical safety feature for medical AI.

## Installation
```bash
git clone https://github.com/Zeesky-code/sepsis-rl.git
cd sepsis-rl
pip install torch pandas numpy matplotlib
```

## Usage
1.  **Generate Data**: Create synthetic patient logs.
    ```bash
    python data_generator.py
    ```
2.  **Train Agent**: Train the CQL agent on the offline data.
    ```bash
    python train_offline.py
    ```
3.  **Evaluate**: Compare the AI Agent vs. the Clinician Baseline.
    ```bash
    python evaluate_policy.py
    ```

## Results
The agent learns to balance blood pressure stabilization against the risks of aggressive treatment, recovering a policy comparable to the clinician baseline solely from observation.

## License
MIT
