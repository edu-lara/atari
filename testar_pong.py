import gymnasium as gym
import ale_py

def main():
    gym.register_envs(ale_py)
    env = gym.make("ALE/Pong-v5", render_mode="human")

    try:
        observacao, info = env.reset(seed=42)
        print("Pong iniciado. Executando ações aleatórias.")

        for _ in range(450):
            acao = env.action_space.sample()
            observacao, recompensa, terminou, limite, info = env.step(acao)

            if terminou or limite:
                observacao, info = env.reset()

        print("Teste concluído.")

    except KeyboardInterrupt:
        print("\nTeste interrompido.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
