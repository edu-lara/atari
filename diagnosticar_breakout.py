import ale_py
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack

NUM_PARTIDAS = 30
ARQUIVO_MODELO = "modelos/breakout_atual.zip"


def main():
    gym.register_envs(ale_py)
    modelo = PPO.load(ARQUIVO_MODELO, device="cpu")

    env = make_atari_env(
        "BreakoutNoFrameskip-v4",
        n_envs=1,
        seed=123,
        # Mede a pontuação real do jogo, sem clipping de recompensa.
        wrapper_kwargs={"terminal_on_life_loss": False, "clip_reward": False},
    )
    env = VecFrameStack(env, n_stack=4)

    try:
        print(f"Passos acumulados: {modelo.num_timesteps:,}")
        print(f"Avaliando {NUM_PARTIDAS} partidas em cada modo...", flush=True)

        media_deterministica, desvio_deterministico = evaluate_policy(
            modelo,
            env,
            n_eval_episodes=NUM_PARTIDAS,
            deterministic=True,
        )
        media_probabilistica, desvio_probabilistico = evaluate_policy(
            modelo,
            env,
            n_eval_episodes=NUM_PARTIDAS,
            deterministic=False,
        )

        print("\nRESULTADO DO DIAGNÓSTICO")
        print(
            "Modo determinístico: "
            f"média {media_deterministica:.2f}, "
            f"desvio padrão {desvio_deterministico:.2f}"
        )
        print(
            "Modo probabilístico: "
            f"média {media_probabilistica:.2f}, "
            f"desvio padrão {desvio_probabilistico:.2f}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
