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
        while True:
            limpar_terminal()
            print(f"Qual modelo deseja {acao}?\n")
            print("1. Melhor modelo")
            print("2. Modelo atual (último) — padrão")
            print("0. Voltar")

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
            print("\nOpção inválida. Digite 0, 1 ou 2.\n")
            input("Pressione Enter para continuar.")

    # Importa e carrega o algoritmo somente depois que o usuário escolher o modelo.
    # Isso evita carregar um modelo que não será usado.
    from stable_baselines3 import PPO
    return caminho, PPO.load(caminho, device="cpu")


def escolher_modo(slug):
    deterministico_padrao = slug != "video_pinball"
    modo_padrao = "1" if deterministico_padrao else "2"
    descricao_padrao = "determinístico" if deterministico_padrao else "probabilístico"

    while True:
        limpar_terminal()
        print("Como deseja que o agente escolha as ações?\n")
        print("1. Determinístico (ação mais provável) - padrão")
        print("2. Probabilístico (probabilidades aprendidas)")
        print("0. Voltar")

        escolha = input(
            f"\nEscolha (0-2, Enter para {descricao_padrao}): "
        ).strip() or modo_padrao
        if escolha == "0":
            return None
        if escolha == "1":
            return True
        if escolha == "2":
            return False
        print("\nOpção inválida. Digite 0, 1 ou 2.\n")
        input("Pressione Enter para continuar.")


def main():
    limpar_terminal()
    while True:
        jogo = escolher_jogo("assistir")
        if jogo is None:
            return
        nome, ambiente_id, slug = jogo
        while True:
            modelo_selecionado = escolher_modelo(nome, slug)
            if modelo_selecionado is None:
                break
            arquivo_modelo, modelo = modelo_selecionado
            deterministico = escolher_modo(slug)
            if deterministico is not None:
                break
            # Volta à escolha do modelo; se só houver um, volta ao jogo.
            if len(encontrar_modelos(slug)) == 1:
                break
        if modelo_selecionado is not None and deterministico is not None:
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
            acao, _ = modelo.predict(observacao, deterministic=deterministico)
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
