import json
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from assistir import encontrar_modelos
from jogos import JOGOS
from terminal import limpar_terminal


def ler_passos(caminho):
    """Lê num_timesteps do arquivo ZIP do Stable-Baselines3 sem carregar o modelo."""
    try:
        with ZipFile(caminho) as arquivo:
            dados = json.loads(arquivo.read("data"))
        return dados.get("num_timesteps")
    except (BadZipFile, KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def formatar_passos(passos):
    if passos is None:
        return "—"
    return f"{int(passos):,}".replace(",", ".")


def main():
    limpar_terminal()
    print("Passos de treinamento por jogo\n")

    linhas = []
    for nome, _, slug in JOGOS.values():
        modelos = encontrar_modelos(slug)
        melhor = ler_passos(modelos["melhor"]) if "melhor" in modelos else None
        atual = ler_passos(modelos["atual"]) if "atual" in modelos else None
        linhas.append((nome, formatar_passos(melhor), formatar_passos(atual)))

    largura_jogo = max(len("Jogo"), *(len(linha[0]) for linha in linhas))
    cabecalho = f"{'Jogo':<{largura_jogo}}  {'Melhor modelo':>14}  {'Modelo atual':>14}"
    print(cabecalho)
    print("-" * len(cabecalho))
    for nome, melhor, atual in linhas:
        print(f"{nome:<{largura_jogo}}  {melhor:>14}  {atual:>14}")

    input("\nPressione Enter para voltar ao menu.")


if __name__ == "__main__":
    main()
