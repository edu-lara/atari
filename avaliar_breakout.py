from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)
NUM_PARTIDAS = 30


class AgenteAleatorio:
    def __init__(self, espaco_acoes):
        self.espaco_acoes = espaco_acoes
        self.espaco_acoes.seed(123)

    def predict(
        self, observation, state=None,
        episode_start=None, deterministic=True
    ):
        acao = np.array([self.espaco_acoes.sample()])
        return acao, state


def avaliar(nome, modelo=None):
    env = make_atari_env(
        "BreakoutNoFrameskip-v4",
        n_envs=1,
        seed=123,
        # A avaliação usa a pontuação real do jogo, sem o clipping do treino.
        wrapper_kwargs={"terminal_on_life_loss": False, "clip_reward": False},
    )
    env = VecFrameStack(env, n_stack=4)

    try:
        agente = modelo if modelo is not None else AgenteAleatorio(
            env.action_space
        )

        print(f"\nAvaliando {nome}: {NUM_PARTIDAS} partidas...", flush=True)

        recompensas, _ = evaluate_policy(
            agente,
            env,
            n_eval_episodes=NUM_PARTIDAS,
            deterministic=True,
            return_episode_rewards=True,
        )

        media = float(np.mean(recompensas))
        desvio = float(np.std(recompensas))
        partidas_positivas = sum(r > 0 for r in recompensas)

        print(f"Resultados: {recompensas}")
        print(f"Recompensa média: {media:.2f}")
        print(f"Desvio padrão: {desvio:.2f}")
        print(f"Partidas com pontuação positiva: {partidas_positivas}/{NUM_PARTIDAS}")
        return media

    finally:
        env.close()


def main():
    melhor = Path("breakout_melhor/best_model.zip")
    modelo_atual = Path("modelos/breakout_atual.zip")
    arquivo_modelo = melhor if melhor.exists() else modelo_atual
    modelo = PPO.load(arquivo_modelo, device="cpu")
    print(f"Avaliando modelo: {arquivo_modelo}")
    print(f"Passos acumulados no modelo: {modelo.num_timesteps:,}")

    media_aleatorio = avaliar("agente aleatório")
    media_treinado = avaliar("agente treinado", modelo)

    print("\nCOMPARAÇÃO FINAL")
    print(f"Agente aleatório: {media_aleatorio:.2f}")
    print(f"Agente treinado: {media_treinado:.2f}")
    print(f"Diferença: {media_treinado - media_aleatorio:+.2f}")


if __name__ == "__main__":
    main()
