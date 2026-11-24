import os
import select
import sys

try:
    import msvcrt
except ImportError:
    msvcrt = None

try:
    import termios
    import tty
except ImportError:
    termios = None
    tty = None


def limpar_terminal():
    if sys.stdout.isatty():
        os.system("cls" if os.name == "nt" else "clear")


def aguardar_continuar():
    if sys.stdin.isatty():
        input("\nConcluído. Pressione Enter para continuar.")


def garantir_terminal_interativo():
    """Reativa eco e leitura por linha após um subprocesso interativo."""
    if not sys.stdin.isatty() or termios is None:
        return

    fd = sys.stdin.fileno()
    configuracao = termios.tcgetattr(fd)
    configuracao[3] |= termios.ECHO | termios.ICANON | termios.ISIG
    termios.tcsetattr(fd, termios.TCSADRAIN, configuracao)


class LeitorTeclaQ:
    def __init__(self):
        self.fd = None
        self.configuracao_original = None

    def iniciar(self):
        if os.name == "nt" and msvcrt is not None:
            return
        if sys.stdin.isatty() and termios is not None and tty is not None:
            self.fd = sys.stdin.fileno()
            self.configuracao_original = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)

    def pressionada(self):
        if os.name == "nt" and msvcrt is not None and sys.stdin.isatty():
            while msvcrt.kbhit():
                if msvcrt.getwch().lower() == "q":
                    return True
        elif self.fd is not None:
            pronto, _, _ = select.select([self.fd], [], [], 0)
            if pronto and b"q" in os.read(self.fd, 64).lower():
                return True
        return False

    def restaurar(self):
        if self.configuracao_original is not None:
            termios.tcsetattr(
                self.fd,
                termios.TCSADRAIN,
                self.configuracao_original,
            )
            self.configuracao_original = None


