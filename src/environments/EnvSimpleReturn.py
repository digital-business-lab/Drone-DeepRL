""" 
This file holds a environment called EnvSimpleReturn. As the name
says, the agent has to reach B and then continue back to A.
"""
import math
import gymnasium as gym
import pybullet as p
import pybullet_data
import time
import numpy as np

class EnvSimpleReturn(gym.Env):
    def __init__(self):
        super().__init__()

        # Setup PyBullet
        self.g_force = 10
        self.physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version # p.GUI
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -self.g_force)  # Gravity along x, y, z

        # Load objects
        self.plane_id = p.loadURDF("plane.urdf", globalScaling=10)
        self.drone_id = p.loadURDF("C:/Git_Repos/Drone-DeepRL/res/drone.urdf", basePosition=[0, 1, 0.8])
        self.drone_initial_pos, self.drone_initial_ori = p.getBasePositionAndOrientation(self.drone_id)

        # Target positions of the cubes to calculate distance between drone and cube
        self.target_a, self.target_b = self.create_checkpoints()
        self.current_target = self.target_b  # Fly to B first
        self.reached_target_b = False
        self.reward_target_b = 100

        # Get prameters for step
        # Weigh drone and write correct mass in drone.urdf
        self.mass = p.getDynamicsInfo(self.drone_id, -1)[0]
        self.thrust_hover = self.mass * self.g_force / 4

        # Drone Rotor Positions (Relative to Center)
        self.motor_positions = [
            [0.2, 0.2, 0],  # Front-left
            [-0.2, 0.2, 0],  # Front-right
            [0.2, -0.2, 0],  # Back-left
            [-0.2, -0.2, 0],  # Back-right
        ]

        # Action space
        self.action_space = gym.spaces.Discrete(6) # Up, Down, Left, Right, Forward, Backward
        self.observation_space = gym.spaces.Box(
            low=np.array([-100, -100, 0.2, 0, 0]),
            high=np.array([100, 100, 10, 50, 1]),
            dtype=np.float32
        )

        # Reward
        self.previous_distance = math.sqrt(
            (self.target_a[0] - self.target_b[0]) ** 2 +  # X difference squared
            (self.target_a[1] - self.target_b[1]) ** 2 +  # Y difference squared
            (self.target_a[2] - self.target_b[2]) ** 2    # Z difference squared
        )

        self.standard_dist = math.sqrt(
            (self.target_a[0] - self.target_b[0]) ** 2 +  # X difference squared
            (self.target_a[1] - self.target_b[1]) ** 2 +  # Y difference squared
            (self.target_a[2] - self.target_b[2]) ** 2    # Z difference squared
        )

    def create_checkpoints(self):
        """ Creates two checkpoint cubes. """
        col_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2])
        cube_mass = 0

        obj_id_a = p.createMultiBody(
            baseMass=cube_mass,
            baseCollisionShapeIndex=col_shape_id,
            basePosition=[0, 0, 0.25]
        )

        obj_id_b = p.createMultiBody(
            baseMass=cube_mass,
            baseCollisionShapeIndex=col_shape_id,
            basePosition=[0, 12, 0.25]
        )

        pos_a, _ = p.getBasePositionAndOrientation(obj_id_a)
        pos_b, _ = p.getBasePositionAndOrientation(obj_id_b)

        return pos_a, pos_b

    def step(self, action: int):
        """ Keep the drone hovering at the target height. """
        z_scaler = 1 # Force on z
        y_scaler = 0 # Force on y
        x_scaler = 0 # Force on x

        # Define actions
        if action == 0: # Up
            z_scaler = 200.5
        elif action == 1: # Right
            x_scaler = 200.5
        elif action == 2: # Down
            z_scaler = -101.5
        elif action == 3: # Left
            x_scaler = -200.5
        elif action == 4: # Forward
            y_scaler = 200.5
        elif action == 5: # Backward
            y_scaler = -200.5
        else:
            raise ValueError(f"Action not in range(6) '{action}'")

        # Apply Force -> Fly
        for motor_pos in self.motor_positions:
            p.applyExternalForce(
                self.drone_id, 
                -1, 
                [x_scaler, y_scaler, self.thrust_hover + z_scaler], 
                motor_pos, 
                p.LINK_FRAME
                )
            
        # Step simulation 
        p.stepSimulation()
        time.sleep(1 / 240)

        obs = self.get_observation() # x, y, z, distance to target -> Drone position
        reward, done, terminated, truncated = self.calc_reward()

        return obs, reward, terminated, truncated, {}

    def reset(self, seed: int =42):
        np.random.seed(seed=seed)
        p.resetBasePositionAndOrientation(
            self.drone_id,
            self.drone_initial_pos, 
            self.drone_initial_ori
        )

        self.current_target = self.target_b
        self.reached_target_b = False
        self.reward_target_b = 100
        return self.get_observation(), {}

    def calc_reward(self):
        reward = 0
        done = False
        truncated = False
        terminated = False

        # Get drone position and calculate distance to the current target
        pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        distance = np.linalg.norm(np.array(pos) - self.current_target)

        # Normalize distance (scale between 0 and 1)
        distance_norm = distance / self.standard_dist  # 1 = farthest, 0 = reached target

        # **Shaped reward: Reward for moving closer (using normalized distance)**
        reward += (self.previous_distance - distance) * 10  # Difference in distance

        # **Extra penalty for being farther away** (scaled with normalized distance)
        reward -= distance_norm * 50  # Larger penalty when far

        # **Crash Penalties (too high or too low)**
        if pos[2] > 2 or pos[2] < 0.2:  
            reward -= 200
            done = True
            truncated = True

        # **Hard penalty if drone moves too far away (out of bounds)**
        if distance > self.standard_dist:
            reward -= 200
            done = True
            truncated = True

        # **Reward for reaching the target**
        if distance < 1:
            if self.reached_target_b:
                print("DONE!")
                reward += 300  # Final goal reached
                done = True
                terminated = True
            else:
                print("Reached target B")
                self.reached_target_b = True
                self.current_target = self.target_a
                reward += self.reward_target_b
                self.reward_target_b = 0  # Prevent repeated rewards

        # **Update previous distance for the next step**
        self.previous_distance = distance
        return reward, done, terminated, truncated

    def get_observation(self):
        pos, _ = p.getBasePositionAndOrientation(self.drone_id)

        # Include distance to target into obeservation
        distance = np.linalg.norm(np.array(pos) - self.current_target)
        drone_id = 0 if self.current_target == self.target_b else 1
        return np.array([pos[0], pos[1], pos[2], distance, drone_id], dtype=np.float32)

    # Only for debugging
    def step_simulation(self, steps=1000):
        """ Run the simulation for a given number of steps while stabilizing the drone. """
        for _ in range(steps):
            self.step(4)
            time.sleep(10)

if __name__ == "__main__":
    env = EnvSimpleReturn()
    env.step_simulation()
