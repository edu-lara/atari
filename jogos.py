JOGOS = {
    "1": ("Freeway", "FreewayNoFrameskip-v4", "freeway"),
    "2": ("Atlantis", "AtlantisNoFrameskip-v4", "atlantis"),
    "3": ("Video Pinball", "VideoPinballNoFrameskip-v4", "video_pinball"),
    "4": ("Pong", "PongNoFrameskip-v4", "pong"),
    "5": ("Breakout", "BreakoutNoFrameskip-v4", "breakout"),
}


def escolher_jogo(acao="executar"):
    print(f"Qual jogo deseja {acao}?")
    for numero, (nome, _, _) in JOGOS.items():
        print(f"{numero}. {nome}")
    print("0. Voltar")

    while True:
        escolha = input("Escolha (0-5): ").strip()
        if escolha == "0":
            return None
        if escolha in JOGOS:
            return JOGOS[escolha]
        print("Opção inválida. Digite um número de 0 a 5.")
