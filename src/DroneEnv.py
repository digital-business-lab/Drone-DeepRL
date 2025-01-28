""" 
Currently this file serves as a test script to experiment with
pybullet / gymnasium and get the perfect environment for our 
use case

Use Case:
We want a roboter which controls a drone from an external device
(mobile phone) which receives the environment via video streaming
and the controlls to the drone

We have to defide this use case in small steps and the first step
is to create an environment with two checkpoints (a, b) and a drone.
The drone has to fly from a to b, wait there for 10 seconds and return
back to a. We will use DRL.

@author Lukas Graf
"""
import time
import random

import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces
import pybullet_data

class DroneEnv(gym.Env):
    # Maybe for the drone we have to create a custom Model (URDF)
    # http://wiki.ros.org/urdf/Tutorials/Create%20your%20own%20urdf%20file
    def __init__(self):
        super().__init__()
        # Action Space: [Throttle (Z), X-force, Y-force]
        self.action_space = spaces.Box(
            low=np.array([-10, -10, -10]), high=np.array([10, 10, 10]), dtype=np.float32
        )

        # Observation Space: [x, y, z, vx, vy, vz]
        self.observation_space = spaces.Box(
            low=np.array([-100, -100, 0, -10, -10, -10]),
            high=np.array([100, 100, 100, 10, 10, 10]),
            dtype=np.float32,
        )
        
        # Setup pybullet
        self.physicsClient = p.connect(p.GUI) # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10) # Gravity along x, y, z
        # Load objects
        housing_id = p.loadURDF("samurai.urdf")
        self.drone_id = p.loadURDF("./objects/drone.urdf", basePosition=[-7, 3, 2])
        self.drone_initial_pos, self.drone_initial_ori = p.getBasePositionAndOrientation(self.drone_id)
        # Target positions of the cubes to calculate distance between drone and cube
        self.target_a, self.target_b = self.create_checkpoints()
        self.current_target = self.target_b # Fly to B first



        # num_joints_drone = p.getNumJoints(self.drone_id)
        # joint_dict_drone = {
        #     idx : p.getJointInfo(self.drone_id, idx)[1] for idx in range(num_joints_drone)
        #     }
        # print(joint_dict_drone)
        # for i in range(1000):
        #     self.step([20, -5, 0])
        # # For testing
        # time.sleep(5)
        # p.disconnect()



    def create_checkpoints(self) -> None:
        # Creates a block, can be use for checkpoint a and b
        col_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents = [0.2, 0.2, 0.2])
        cube_mass = 0

        obj_id_a = p.createMultiBody(
            baseMass = cube_mass,
            baseCollisionShapeIndex = col_shape_id,
            basePosition = [0, 0, 0.25]
        )

        obj_id_b = p.createMultiBody(
            baseMass = cube_mass,
            baseCollisionShapeIndex = col_shape_id,
            basePosition = [-15, 10, 2.25]
        )

        pos_a, _ = p.getBasePositionAndOrientation(obj_id_a)
        pos_b, _ = p.getBasePositionAndOrientation(obj_id_b)

        return pos_a, pos_b
            
    def reset(self, seed=42, **kwargs):
        # Optionally set the seed for reproducibility
        np.random.seed(seed)

        # Reset the drone to its initial position
        p.resetBasePositionAndOrientation(
            self.drone_id, 
            self.drone_initial_pos, 
            self.drone_initial_ori
        )
        
        # Set the current target to be checkpoint B
        self.current_target = self.target_b
        self.time_flying = 0
        # Return the initial observation and an empty dictionary for info
        return self.get_observation(), {}
    
    def step(self, action: list):
        # Scale action to apply realistic forces
        thrust = action[0] * 10  # Z-axis thrust
        force_x = action[1] * 10  # X-axis force
        force_y = action[2] * 10  # Y-axis force

        # Apply forces to the drone
        p.applyExternalForce(self.drone_id, -1, [force_x, force_y, thrust], [0, 0, 0], p.LINK_FRAME)

        # Step simulation
        p.stepSimulation()
        time.sleep(1 / 240)

        self.time_flying += 1 / 240
        if self.time_flying > 10:
            print("Drone took too long! Resetting drone.")
            self.reset()
            
        # Compute observations, reward, and done status
        obs = self.get_observation()
        reward, done = self.compute_reward_done()

        # Return the five values expected by stable-baselines3
        terminated = False  # We don't have a specific termination condition, so this will be False
        truncated = False  # We can use this flag if there is a time limit or episode limit
        info = {}

        return obs, reward, done, truncated, info

    def get_observation(self):
        # Get drone position and velocity
        pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        vel, _ = p.getBaseVelocity(self.drone_id)
        return np.array(pos + vel)
    
    def compute_reward_done(self):
        pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        distance = np.linalg.norm(np.array(pos) - self.current_target)

        # Reward is negative distance to the target (the closer, the better)
        reward = -distance

        # Bonus for reaching the target, but not excessively large
        if distance < 0.5:
            if np.allclose(self.current_target, self.target_b):
                self.current_target = self.target_a
            else:
                self.current_target = self.target_b
            reward += 50  # More moderate bonus for reaching the target
            done = True
        else:
            done = False

        # Add additional reward when the drone is above x = 0
        if pos[2] > 0:
            reward += 10  # Give a reward for being above x = 0

        return reward, done
    
    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect()



if __name__ == "__main__":
    env = DroneEnv()
    # if -1 -> not connected
    print(env.physicsClient)

