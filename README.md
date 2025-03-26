# Drone-DeepRL
The project goal is that we can automatically fly a drone from the starting point A to the goal point B. On B the drone should pickup some kind of object and fly it back to the starting point. To achive this the main idea is to make a RL algorithm to fly the drone. Because of inefficiency of learning in the real world we create environments via pybullet so the algorithm learns to avoid objects, fly to the desired points and make it quick. After that we can run the algorithm on the drone and test it in real life.


### Environment
---
- **EnvRandomReturn.py** -> An environment written in pybullet that has the goal that the agent flies from point A to a random generated point B and return to A

- **EnvSimpleLine.py** -> An environemt written in pybullet that has the goal that the agent flies from a fixed point A to a fixed point B

- **EnvSimpleReturn.py** -> An environment written in pybullet that has the goal that the agent flies from a fixed point A to a fixed point B and return to A

### Models
---
- **Model.py** -> This file contains the models PPO and DQN which are state-of-the-art Model Free Reinforcement Learning Algorithms. You can train and test the models with this file.

- **custom_model.ipynb** -> This Jupyter Notebook holds a custom Model-Based RL Model. The main idea behind it is, that we have a world model (LSTM) which learns to predict the observations (next_state, reward, terminated, truncated). Based on this predictions a DQN is trained. In Theory this approach results in faster convergion with the same or even higher asymptotic performance than the model free approaches. If the notebook doesnt run when clicking "Run All" try "Restart" and then "Run all" -> It should work now

### URDF Models
---
- **drone.urdf** -> A custom written model of a drone. This drone is used as the agent in the environment

### Next Steps
---
- Try to make custom model work on custom environments. The model works exremly well on existing CartPole environment in gym (>50 episodes)

- Try out better reward shaping for the Model free algorithms so maybe these also work pretty fast with high asymptotic performance

- Find a way so the agent generalizes better -> The models need a perfect understanding between the current target its flying to and the distance there is between them