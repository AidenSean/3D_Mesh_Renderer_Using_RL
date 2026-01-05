# Pixel RL 3D Builder

**Team Neural Mavericks**

## Overview
**Pixel RL 3D Builder** is an intelligent system that creates 3D structures from natural language commands. It combines **Generative AI** for imagination and **Reinforcement Learning** for execution.

![Concept](assets/logo.png) (If you have a logo, place it here)

## How It Works
1.  **Speech Command**: The user says "Build a blue car".
2.  **Blueprint Generation**: A Latent Consistency Model (LCM) generates a 2D pixel art blueprint.
3.  **3D Estimation**: Depth-Anything converts the 2D image into a 3D voxel grid.
4.  **Construction**: An RL Agent (or Heuristic Agent) navigates the grid and places blocks to match the blueprint.
5.  **Output**: The final result is smoothed into a standard `.obj` 3D mesh file.

## Installation
1.  Clone the repository:
    ```bash
    git clone <your-repo-url>
    cd pixel_rl_3d_builder
    ```
2.  Create virtual environment:
    ```bash
    python -m venv rl_env
    source rl_env/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
Run the main application:
```bash
./run.sh
```
Or manually:
```bash
python main.py
```

## Structure
*   `src/`: Core source code (Agent, Environment, Visualizer).
*   `assets/`: Generated 3D objects.
*   `models/`: Saved RL models.
