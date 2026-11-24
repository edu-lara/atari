import ale_py
import gymnasium as gym
import subprocess
import sys
from pathlib import Path
from terminal import garantir_terminal_interativo, limpar_terminal

# Os workers do SubprocVecEnv usam spawn e reimportam este módulo principal.
# Registrar aqui deixa os ambientes ALE disponíveis também nesses processos.
gym.register_envs(ale_py)


def main():
    opcoes = {
        "1": ("Treinar", "treinar"),
        "2": ("Assistir", "assistir"),
        "3": ("Avaliar", "avaliar"),
        "4": ("Diagnosticar", "diagnosticar"),
        "5": ("Simular ambiente", "simular"),
        "6": ("Ver progresso do treinamento", "progresso"),
    }

    while True:
        limpar_terminal()
        print("O que deseja fazer?\n")
        for numero, (nome, _) in opcoes.items():
            print(f"{numero}. {nome}")
        print("0. Sair")

        escolha = input("\nEscolha (0-6): ").strip()
        if escolha == "0":
            print("Saindo.")
            return
        if escolha not in opcoes:
            print("\nOpção inválida. Digite um número de 0 a 6.\n")
            input("Pressione Enter para continuar.")
            continue

        _, modulo = opcoes[escolha]
        if modulo in ("assistir", "simular", "progresso"):
            arquivo_utilitario = Path(__file__).resolve().with_name(f"{modulo}.py")
            try:
                subprocess.run(
                    [sys.executable, str(arquivo_utilitario)],
                    cwd=Path(__file__).resolve().parent,
                    check=False,
                )
            except KeyboardInterrupt:
                print("\nOperação interrompida.")
            finally:
                garantir_terminal_interativo()
            continue

        if modulo == "treinar":
            from treinar import main as executar
        elif modulo == "avaliar":
            from avaliar import main as executar
        elif modulo == "diagnosticar":
            from diagnosticar import main as executar

        executar()


if __name__ == "__main__":
    main()
