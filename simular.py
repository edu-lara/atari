import ale_py
import gymnasium as gym

from jogos import escolher_jogo
from terminal import LeitorTeclaQ, limpar_terminal


def main():
    limpar_terminal()
    jogo = escolher_jogo("simular")
    if jogo is None:
        return
    nome, ambiente_id, _ = jogo

    gym.register_envs(ale_py)
    nome_ale = ambiente_id.replace("NoFrameskip-v4", "")
    env = gym.make(f"ALE/{nome_ale}-v5", render_mode="human")
    teclado = LeitorTeclaQ()
    try:
        observacao, info = env.reset(seed=42)
        print(f"Ambiente: {nome}")
        print("Simulação com ações aleatórias: até 450 passos.")
        print("Pressione Q neste terminal para sair.")
        teclado.iniciar()
        for _ in range(450):
            acao = env.action_space.sample()
            observacao, recompensa, terminou, limite, info = env.step(acao)
            if teclado.pressionada():
                print("\nSimulação interrompida.")
                break
            if terminou or limite:
                observacao, info = env.reset()
        else:
            print("Simulação concluída.")
    except KeyboardInterrupt:
        print("\nSimulação interrompida.")
    finally:
        teclado.restaurar()
        env.close()


if __name__ == "__main__":
    main()
