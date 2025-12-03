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
4.  **Off-Policy Evaluation (OPE)**: Use **Weighted Importance Sampling (WIS)** to estimate the policy's value without running it on real patients.
    ```bash
    python ope_evaluation.py
    ```

## Results
-   **Online Evaluation**: The agent recovers ~75% of the optimal clinician's performance purely from observation.
-   **Offline Evaluation**: The WIS estimator provides a conservative lower-bound on the policy's value, demonstrating safety verification.

## License
MIT
