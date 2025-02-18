# Drone-DeepRL

### Status
---
08.02.2025
- Created an environment for the drone to fly from point A to B
- We tried out a model-free algorithm (PPO) which solves this environment
- Ideas: Maybe try out model-based algorithm for faster convergence (https://arxiv.org/abs/1805.12114) 

18.02.2025 
- Model can fly from A to B now -> Used PPO
- Ideas: Try DQN
- TODO: After reaching Point A make the Drone return to B and end episode
<video controls src="doc/PPO_EnvSimpleLine_400k.mp4" title="Title"></video>