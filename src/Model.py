from stable_baselines3 import PPO
from stable_baselines3 import DQN
from environments.EnvSimpleLine import EnvSimpleLine
from environments.EnvSimpleReturn import EnvSimpleReturn
from stable_baselines3.common.env_util import make_vec_env


class PPO_Algorithm:
    def __init__(self, env, timesteps: int, model: str =None):
        self.env = make_vec_env(env, n_envs=1)
        if model:
            self.model = PPO.load(
                model, 
                env=self.env, 
                gamma=0.99, 
                learning_rate=0.001, 
                clip_range=0.2,
                n_batch=2048, 
                n_epochs=10, 
                ent_coef=0.01, 
                vf_coef=0.5, 
                max_grad_norm=0.5
            )
        else:
            self.model = PPO(
                "MlpPolicy", 
                env=self.env, 
                gamma=0.99, 
                learning_rate=0.001, 
                clip_range=0.2,
                batch_size=2048, 
                n_epochs=10, 
                ent_coef=0.01, 
                vf_coef=0.5, 
                max_grad_norm=0.5
            )
        self.timesteps = timesteps

    def train(self, model_name: str):
        self.model.learn(total_timesteps=self.timesteps)
        self.model.save(f"./models/PPO_{model_name}")

    def test(self, epochs: int):
        for episode in range(epochs):
            terminated = False
            obs = self.env.reset()
            while not terminated:
                action, _states = self.model.predict(obs)
                obs, reward, terminated, truncated = self.env.step(action)


class DQN_Algorithm:
    def __init__(self, env, timesteps: int, model: str =None):
        self.env = make_vec_env(env, n_envs=1)
        if model:
            self.model = DQN.load(
                model,  # Path to the saved model
                env=self.env,  # Your environment
                gamma=0.99,
                learning_rate=0.001,
                buffer_size=100000,
                exploration_fraction=0.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.02,
                train_freq=4,
                batch_size=32,
                verbose=1  # Optional: Set to 1 for logging output
            )
        else:
            self.model = DQN(
                "MlpPolicy",  # The policy network (MLP for fully connected network)
                env=self.env,  # Your environment
                gamma=0.99,
                learning_rate=0.001,
                buffer_size=100000,
                exploration_fraction=0.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.02,
                train_freq=4,
                batch_size=32,
                verbose=1  # Optional: Set to 1 for logging output
            )
        self.timesteps = timesteps

    def train(self, model_name: str):
        self.model.learn(total_timesteps=self.timesteps)
        self.model.save(f"./models/RainbowDQN_{model_name}")
    
    def test(self, epochs: int):
        for episode in range(epochs):
            terminated = False
            obs = self.env.reset()
            while not terminated:
                action, _states = self.model.predict(obs)
                obs, reward, terminated, truncated = self.env.step(action)


if __name__ == "__main__":
    # algorithm = PPO_Algorithm(
    #     env=EnvSimpleLine,
    #     model="models/PPO_EnvSimpleLine_400k",
    #     timesteps=400_000
    #     )
    # algorithm.test(epochs=10)

    algorithm = DQN_Algorithm(
        env=EnvSimpleReturn,
        timesteps=400_000
    )
    algorithm.train(model_name="EnvSimpleReturn_400k")
