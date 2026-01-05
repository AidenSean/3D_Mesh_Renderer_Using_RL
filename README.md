# Deep Reinforcement Learning Final Project

**Team Neural Mavericks**

This repository contains two distinct Deep Reinforcement Learning projects:
1.  **Pixel RL 3D Builder:** A custom Embodied AI system that builds 3D models from speech.
2.  **Atari Breakout RL:** A comparative study of 5+ RL algorithms (DQN, PPO, A2C, etc.) on Atari Breakout.

---

## Project 1: Pixel RL 3D Builder
**"From Speech to 3D Voxel Worlds"**

### Overview
This system combines **Generative AI** (Latent Consistency Models) with **Reinforcement Learning** to create an autonomous agent capable of constructing 3D objects based on natural language commands.
*   **Input:** "Build a blue castle."
*   **Process:** Speech -> Text -> 2D Image (GenAI) -> 3D Voxel Blueprint (DepthAI) -> RL Agent Construction.
*   **Output:** A smoothed `.obj` 3D mesh file.

### Installation
1.  Create virtual environment:
    ```bash
    python -m venv rl_env
    source rl_env/bin/activate
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage
Run the main application (with GUI):
```bash
python main.py
```
*   The system will listen for a voice command.
*   Napari viewer will open to show the agent building in real-time.
*   Final mesh is saved to `assets/3d_objects/`.

---

## Project 2: Atari Breakout RL Agents
**"Comparative Study of DRL Algorithms"**

### Overview
Located in `part1_atari_breakout/Codes/`, this section contains clean, single-file implementations of major DRL algorithms trained on `ALE/Breakout-v5`.

**Algorithms Implemented:**
*   **DQN** (Deep Q-Network)
*   **Double DQN**
*   **Dueling DQN**
*   **A2C** (Advantage Actor-Critic)
*   **PPO** (Proximal Policy Optimization)
*   **SAC** (Soft Actor-Critic - adapted for discrete)

### Usage
To train an agent (e.g., DQN), run the specific script from the root directory:

```bash
# Example: Train DQN
python part1_atari_breakout/Codes/dqn_atari.py --env-id ALE/Breakout-v5 --track

# Example: Train PPO
python part1_atari_breakout/Codes/ppo_atari.py --env-id ALE/Breakout-v5
```

**Key Arguments:**
*   `--track`: Enable Weights & Biases logging (Optional).
*   `--capture-video`: Save MP4 videos of the agent.
*   `--seed 1`: Set random seed.

### Results
Check the `part1_atari_breakout/` folder for:
*   `Graph Plots/`: Learning curves.
*   `Videos/`: Replays of trained agents.
*   `models/`: Saved model weights.

---

## Repository Structure
*   `src/`: Source code for **Pixel RL 3D Builder**.
*   `part1_atari_breakout/`: Source code and data for **Atari Project**.
*   `assets/`: Generated images and 3D meshes.
*   `project_report.tex`: Full Academic Report (LaTeX).
*   `simple_summary.tex`: Simplified Project Explanation.
