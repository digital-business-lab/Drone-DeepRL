import gymnasium as gym
import pybullet as p
import pybullet_data
import time
import numpy as np

class EnvSimpleLine(gym.Env):
    def __init__(self):
        super().__init__()

        # Setup PyBullet
        self.g_force = 10
        self.physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -self.g_force)  # Gravity along x, y, z

        # Load objects
        self.plane_id = p.loadURDF("plane.urdf", globalScaling=10)
        self.drone_id = p.loadURDF("objects/drone.urdf", basePosition=[0, 4, 1])
        self.drone_initial_pos, self.drone_initial_ori = p.getBasePositionAndOrientation(self.drone_id)

        # Target positions of the cubes to calculate distance between drone and cube
        self.target_a, self.target_b = self.create_checkpoints()
        self.current_target = self.target_b  # Fly to B first

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
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

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
            basePosition=[0, 30, 0.25]
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
            z_scaler = 5.5
        elif action == 1: # Right
            x_scaler = 5.5
        elif action == 2: # Down
            z_scaler = -5.5
        elif action == 3: # Left
            x_scaler = -5.5
        elif action == 4: # Forward
            y_scaler = 5.5
        elif action == 5: # Backward
            y_scaler = -5.5
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

        obs = self.get_observation() # x, y, z -> Drone position
        reward, done, terminated, truncated = self.calc_reward()

        return obs, reward, done, terminated, truncated, {}

    def reset(self, seed: int =42):
        np.random.seed(seed=seed)
        p.resetBasePositionAndOrientation(
            self.drone_id,
            self.drone_initial_pos, 
            self.drone_initial_ori
        )

        self.current_target = self.target_b
        return self.get_observation()

    def calc_reward(self):
        reward = 0
        done = False
        truncated = False
        terminated = False

        # Get current position and distance
        pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        distance = np.linalg.norm(np.array(pos) - self.current_target)

        # Negative distance so it tries to get closer
        reward -= distance

        # Bonus for staying in a specific hight
        if pos[2] > 0.5 and pos[2] < 10:
            reward + 5

        if pos[2] < 0.2: #Crash
            reward - 100
            done = True
            truncated = True

        # Bonus for reaching target
        if distance < 0.5:
            reward + 100
            done = True
            terminated = True

        return reward, done, terminated, truncated

    def get_observation(self):
        pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        return np.array([pos[0], pos[1], pos[2]], dtype=np.float32)

    # Only for debugging
    def step_simulation(self, steps=1000):
        """ Run the simulation for a given number of steps while stabilizing the drone. """
        for _ in range(steps):
            self.step(4)

if __name__ == "__main__":
    from stable_baselines3 import PPO

    env = EnvSimpleLine()
    # env.step_simulation(10000) # Only for debugging

    # Create PPO model
    model = PPO("MlpPolicy", env, verbose=1)

    # Training loop with episode tracking
    num_episodes = 0
    total_timesteps = 4_000_000
    timesteps_per_episode = 40_000

    for timestep in range(0, total_timesteps, timesteps_per_episode):
        # Track episode count manually
        num_episodes += 1
        print(f"Episode {num_episodes} started")
        
        # Reset the environment to start a new episode
        obs = env.reset()

        # Train for this episode
        done = False
        episode_timesteps = 0

        while not done and episode_timesteps < timesteps_per_episode:
            action, _states = model.predict(obs)
            obs, reward, done, terminated, truncated, _ = env.step(action)
            episode_timesteps += 1
        print(f"Episode {num_episodes} completed")

    model.save("drone_rl_final")


