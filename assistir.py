import gc
from pathlib import Path

from ambientes_atari import criar_ambiente_atari
from jogos import escolher_jogo
from terminal import LeitorTeclaQ, limpar_terminal


def encontrar_modelos(slug):
    melhor = Path("modelos") / f"{slug}_melhor" / "best_model.zip"
    if not melhor.exists():
        # Compatibilidade com a localização usada por versões anteriores.
        melhor = Path(f"{slug}_melhor") / "best_model.zip"

    atual = Path("modelos") / f"{slug}_atual.zip"
    return {
        nome: caminho
        for nome, caminho in (("melhor", melhor), ("atual", atual))
        if caminho.exists()
    }


def escolher_modelo(nome, slug, acao="assistir"):
    caminhos = encontrar_modelos(slug)
    if not caminhos:
        print(f"Nenhum modelo de {nome} foi encontrado.\n")
        return None

    if len(caminhos) == 1:
        caminho = next(iter(caminhos.values()))
    else:
        print(f"\nQual modelo deseja {acao}?")
        print("1. Melhor modelo")
        print("2. Modelo atual (último) — padrão")
        print("0. Voltar")

        while True:
            escolha = input("\nEscolha (0-2, Enter para 2): ").strip() or "2"
            if escolha == "0":
                limpar_terminal()
                return None
            if escolha == "1":
                caminho = caminhos["melhor"]
                break
            if escolha == "2":
                caminho = caminhos["atual"]
                break
            print("Opção inválida. Digite 0, 1 ou 2.")

    # Importa e carrega o algoritmo somente depois que o usuário escolher o modelo.
    # Isso evita carregar um modelo que não será usado.
    from stable_baselines3 import PPO
    return caminho, PPO.load(caminho, device="cpu")


def main():
    limpar_terminal()
    while True:
        jogo = escolher_jogo("assistir")
        if jogo is None:
            return
        nome, ambiente_id, slug = jogo
        modelo_selecionado = escolher_modelo(nome, slug)
        if modelo_selecionado is not None:
            arquivo_modelo, modelo = modelo_selecionado
            break

    env = criar_ambiente_atari(
        ambiente_id,
        n_envs=1,
        seed=123,
        terminal_on_life_loss=False,
        clip_reward=(slug != "breakout"),
        env_kwargs={"render_mode": "human"},
        iniciar_com_fire=(slug == "video_pinball"),
    )
    teclado = LeitorTeclaQ()

    try:
        observacao = env.reset()
        recompensa_total = 0.0

        if slug == "pong":
            print("O agente controla a raquete da direita.")
        print(f"Modelo carregado: {arquivo_modelo}")
        passos = f"{modelo.num_timesteps:,}".replace(",", ".")
        print(f"Passos de treinamento realizados: {passos}")
        print("Pressione Q neste terminal para sair.")
        teclado.iniciar()

        while True:
            # No Pinball, a ação modal do modelo pode manter uma combinação
            # como RIGHTFIRE indefinidamente; amostrar a política permite que
            # ela alterne flippers e use DOWN/FIRE quando necessário.
            acao, _ = modelo.predict(
                observacao, deterministic=(slug != "video_pinball")
            )
            observacao, recompensas, terminou, _ = env.step(acao)
            recompensa_total += float(recompensas[0])

            if teclado.pressionada():
                print("\nVisualização encerrada.")
                break
            if terminou[0]:
                print(f"Recompensa da partida: {recompensa_total:.0f}")
                break

    except KeyboardInterrupt:
        print("\nVisualização interrompida.")
    finally:
        teclado.restaurar()
        env.close()
        # O ALE encerra a janela nativa quando seu objeto é destruído.
        del env
        gc.collect()


if __name__ == "__main__":
    main()
