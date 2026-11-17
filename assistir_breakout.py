from pathlib import Path

import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


def main():
    gym.register_envs(ale_py)
    env = make_atari_env(
        "BreakoutNoFrameskip-v4",
        n_envs=1,
        seed=123,
        env_kwargs={"render_mode": "human"},
        # Exibe a pontuação real do Breakout ao final da partida.
        wrapper_kwargs={"terminal_on_life_loss": False, "clip_reward": False},
    )
    env = VecFrameStack(env, n_stack=4)

    try:
        melhor = Path("breakout_melhor/best_model.zip")
        modelo_atual = Path("modelos/breakout_atual.zip")
        arquivo_modelo = melhor if melhor.exists() else modelo_atual
        modelo = PPO.load(arquivo_modelo, device="cpu")
        observacao = env.reset()
        recompensa_total = 0.0

        print(f"Modelo de Breakout carregado: {arquivo_modelo}")

        while True:
            acao, _ = modelo.predict(observacao, deterministic=True)
            observacao, recompensas, terminou, _ = env.step(acao)
            recompensa_total += float(recompensas[0])

            if terminou[0]:
                print(f"Recompensa da partida: {recompensa_total:.0f}")
                break

    except KeyboardInterrupt:
        print("\nVisualização interrompida.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
