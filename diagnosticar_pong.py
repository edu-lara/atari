import ale_py
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

modelo = PPO.load("modelos/pong_atual.zip", device="cpu")

env = make_atari_env(
    "PongNoFrameskip-v4",
    n_envs=1,
    seed=123,
    wrapper_kwargs={"terminal_on_life_loss": False},
)
env = VecFrameStack(env, n_stack=4)

try:
    print(f"Passos acumulados: {modelo.num_timesteps:,}")
    print("Avaliando 30 partidas com escolhas probabilísticas...", flush=True)

    media, desvio = evaluate_policy(
        modelo,
        env,
        n_eval_episodes=30,
        deterministic=False,
    )

    print("\nRESULTADO DO DIAGNÓSTICO")
    print("Média da avaliação determinística anterior: -21.00")
    print(f"Média da avaliação probabilística: {media:.2f}")
    print(f"Desvio padrão: {desvio:.2f}")

finally:
    env.close()
