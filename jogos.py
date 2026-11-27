from terminal import limpar_terminal

JOGOS = {
    "1": ("Freeway", "FreewayNoFrameskip-v4", "freeway"),
    "2": ("Atlantis", "AtlantisNoFrameskip-v4", "atlantis"),
    "3": ("Video Pinball", "VideoPinballNoFrameskip-v4", "video_pinball"),
    "4": ("Pong", "PongNoFrameskip-v4", "pong"),
    "5": ("Breakout", "BreakoutNoFrameskip-v4", "breakout"),
    "6": ("Enduro", "EnduroNoFrameskip-v4", "enduro"),

    # Para adicionar um jogo Atari:
    # 1. Consulte https://ale.farama.org/environments/ e abra a página do jogo
    #    para encontrar o ID do ambiente (por exemplo, MsPacmanNoFrameskip-v4).
    # 2. Duplique uma linha e atualize o nome, o ID do ambiente e o slug.

}


def escolher_jogo(acao="executar"):
    while True:
        limpar_terminal()
        print(f"Qual jogo deseja {acao}?\n")
        for numero, (nome, _, _) in JOGOS.items():
            print(f"{numero}. {nome}")
        print("0. Voltar")

        maior_opcao = len(JOGOS)
        escolha = input(f"\nEscolha (0-{maior_opcao}): ").strip()
        if escolha == "0":
            return None
        if escolha in JOGOS:
            return JOGOS[escolha]
        print(f"\nOpção inválida. Digite um número de 0 a {maior_opcao}.\n")
        input("Pressione Enter para continuar.")
