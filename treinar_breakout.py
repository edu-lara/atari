import os
import re
import select
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.utils import FloatSchedule
from ambientes_atari import criar_ambiente_atari

try:
    import msvcrt  # Leitura de teclas sem bloquear, disponível no Windows.
except ImportError:
    msvcrt = None

try:
    import termios  # Configuração do terminal em sistemas Unix.
    import tty
except ImportError:
    termios = None
    tty = None

PASTA_MODELOS = Path("modelos")
PASTA_MELHOR_MODELO = Path("breakout_melhor")
ARQUIVO_MODELO = PASTA_MODELOS / "breakout_atual.zip"
PREFIXO_CHECKPOINT = "breakout"
INTERVALO_CHECKPOINT = 100_000
INTERVALO_AVALIACAO = 250_000
PARTIDAS_POR_AVALIACAO = 20
NUM_AMBIENTES = 4
N_STEPS = 128
BATCH_SIZE = 256
N_EPOCHS = 4
TAXA_APRENDIZADO = 2.5e-4
CLIP_RANGE = 0.1
ENT_COEF = 0.01


def perguntar_passos():
    while True:
        resposta = input("Quantos passos adicionais deseja executar? ").strip()
        try:
            passos = int(resposta)
        except ValueError:
            print("Digite um número inteiro positivo.")
            continue

        if passos > 0:
            return passos
        print("O número de passos deve ser maior que zero.")


def encontrar_checkpoint_mais_recente():
    """Escolhe o checkpoint com maior contador de passos no nome do arquivo."""
    padrao = re.compile(rf"{re.escape(PREFIXO_CHECKPOINT)}_(\d+)_steps\.zip$")
    encontrados = []
    for arquivo in PASTA_MODELOS.glob(f"{PREFIXO_CHECKPOINT}_*_steps.zip"):
        correspondencia = padrao.fullmatch(arquivo.name)
        if correspondencia:
            encontrados.append((int(correspondencia.group(1)), arquivo))
    return max(encontrados, default=(0, None), key=lambda item: item[0])[1]


def aplicar_parametros_atari(modelo):
    """Aplica os parâmetros Atari também ao continuar um modelo existente."""
    modelo.n_steps = N_STEPS
    modelo.batch_size = BATCH_SIZE
    modelo.n_epochs = N_EPOCHS
    modelo.learning_rate = TAXA_APRENDIZADO
    modelo.lr_schedule = FloatSchedule(TAXA_APRENDIZADO)
    modelo.clip_range = FloatSchedule(CLIP_RANGE)
    modelo.ent_coef = ENT_COEF


class AvaliacaoPorPassos(EvalCallback):
    """Dispara avaliações em marcos globais, mesmo entre sessões de treino."""

    def __init__(self, eval_env, intervalo_passos, **kwargs):
        self.intervalo_passos = intervalo_passos
        self.proxima_avaliacao = None
        # O callback pai será chamado apenas quando cruzarmos um marco global.
        super().__init__(eval_env, eval_freq=1, **kwargs)

    def _on_training_start(self):
        super()._on_training_start()
        self.proxima_avaliacao = (
            self.model.num_timesteps // self.intervalo_passos + 1
        ) * self.intervalo_passos

    def _on_step(self):
        if self.num_timesteps < self.proxima_avaliacao:
            return True

        # Avança o marco antes da avaliação para manter a agenda alinhada
        # quando um rollout passar alguns passos do limite solicitado.
        while self.proxima_avaliacao <= self.num_timesteps:
            self.proxima_avaliacao += self.intervalo_passos
        return super()._on_step()


class PararComQ(BaseCallback):
    """Permite parar com Q quando há terminal interativo compatível."""

    def __init__(self):
        super().__init__()
        self.fd = None
        self.configuracao_original = None

    def _on_training_start(self):
        if not sys.stdin.isatty():
            print("Entrada sem terminal interativo. Use Control + C para parar.")
            return

        if os.name == "nt" and msvcrt is not None:
            print("Pressione Q neste terminal para parar e salvar.")
        elif termios is not None and tty is not None:
            try:
                self.fd = sys.stdin.fileno()
                self.configuracao_original = termios.tcgetattr(self.fd)
                tty.setcbreak(self.fd)
                print("Pressione Q neste terminal para parar e salvar.")
            except (OSError, termios.error):
                self.fd = None
                self.configuracao_original = None
                print("Use Control + C para parar e salvar.")
        else:
            print("Use Control + C para parar e salvar.")

    def _on_step(self):
        if os.name == "nt" and msvcrt is not None and sys.stdin.isatty():
            if msvcrt.kbhit() and msvcrt.getwch().lower() == "q":
                print("\nQ recebido. Encerrando e salvando...")
                return False
        elif self.fd is not None:
            pronto, _, _ = select.select([self.fd], [], [], 0)
            if pronto and b"q" in os.read(self.fd, 64).lower():
                print("\nQ recebido. Encerrando e salvando...")
                return False
        return True

    def restaurar_terminal(self):
        # Restaura o modo normal mesmo se o treinamento for interrompido.
        if self.configuracao_original is not None:
            termios.tcsetattr(
                self.fd,
                termios.TCSADRAIN,
                self.configuracao_original,
            )
            self.configuracao_original = None


def main():
    passos_adicionais = perguntar_passos()
    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)

    env = criar_ambiente_atari(
        "BreakoutNoFrameskip-v4",
        # Os ambientes avançam em processos separados para aproveitar os
        # núcleos da CPU enquanto o PPO aprende no processo principal.
        n_envs=NUM_AMBIENTES,
        seed=42,
        terminal_on_life_loss=False,
        subprocess=True,
    )
    env_avaliacao = None

    try:
        # A avaliação usa partidas completas e pontuação real; o treino
        # continua usando recompensa recortada para estabilizar o PPO.
        env_avaliacao = criar_ambiente_atari(
            "BreakoutNoFrameskip-v4",
            n_envs=1,
            seed=123,
            terminal_on_life_loss=False,
            clip_reward=False,
        )

        # Prefere o modelo final; se uma execução foi interrompida antes de
        # salvá-lo, retoma do checkpoint periódico com mais passos treinados.
        checkpoint = encontrar_checkpoint_mais_recente()
        if checkpoint is not None and (
            not ARQUIVO_MODELO.exists()
            or checkpoint.stat().st_mtime > ARQUIVO_MODELO.stat().st_mtime
        ):
            arquivo_para_carregar = checkpoint
            print(f"Retomando do checkpoint: {checkpoint}")
        elif ARQUIVO_MODELO.exists():
            arquivo_para_carregar = ARQUIVO_MODELO
            print("Modelo salvo carregado. Continuando o aprendizado.")
        elif checkpoint is not None:
            arquivo_para_carregar = checkpoint
            print(f"Retomando do checkpoint: {checkpoint}")
        else:
            arquivo_para_carregar = None

        if arquivo_para_carregar is not None:
            modelo = PPO.load(arquivo_para_carregar, env=env, device="cpu")
        else:
            modelo = PPO(
                "CnnPolicy",
                env,
                device="cpu",
                verbose=1,
                seed=42,
                n_steps=N_STEPS,
                batch_size=BATCH_SIZE,
                n_epochs=N_EPOCHS,
                learning_rate=TAXA_APRENDIZADO,
                clip_range=CLIP_RANGE,
                ent_coef=ENT_COEF,
                tensorboard_log="./logs/breakout/",
            )
            print("Criando um modelo novo para Breakout.")

        # O modelo carregado guarda os parâmetros antigos; atualiza-os para
        # que os novos valores também valham nas continuações do treinamento.
        aplicar_parametros_atari(modelo)

        print(f"Incentivo à exploração: ent_coef={modelo.ent_coef}")
        inicio = modelo.num_timesteps
        print(f"Passos acumulados: {inicio:,}")
        print(f"Passos adicionais solicitados: {passos_adicionais:,}")
        print(f"Total previsto: {inicio + passos_adicionais:,}")
        print("O PPO pode passar um pouco do total solicitado para concluir um ciclo.")

        teclado = PararComQ()
        checkpoints = CheckpointCallback(
            # O callback conta iterações vetorizadas, cada uma avançando
            # NUM_AMBIENTES passos. Divide para manter o intervalo em passos.
            save_freq=max(INTERVALO_CHECKPOINT // NUM_AMBIENTES, 1),
            save_path=str(PASTA_MODELOS),
            name_prefix=PREFIXO_CHECKPOINT,
        )
        PASTA_MELHOR_MODELO.mkdir(parents=True, exist_ok=True)
        arquivo_melhor_modelo = PASTA_MELHOR_MODELO / "best_model.zip"
        arquivo_historico = Path("logs/breakout_avaliacao/evaluations.npz")
        avaliacao = AvaliacaoPorPassos(
            env_avaliacao,
            intervalo_passos=INTERVALO_AVALIACAO,
            best_model_save_path=str(PASTA_MELHOR_MODELO),
            log_path="./logs/breakout_avaliacao/",
            n_eval_episodes=PARTIDAS_POR_AVALIACAO,
            deterministic=True,
            verbose=1,
        )

        # Mantém a melhor média entre execuções sucessivas do treinamento.
        if arquivo_historico.exists() and arquivo_melhor_modelo.exists():
            with np.load(arquivo_historico) as historico:
                medias = historico["results"].mean(axis=1)
                if len(medias):
                    avaliacao.best_mean_reward = float(medias.max())
                    avaliacao.evaluations_timesteps = historico["timesteps"].tolist()
                    avaliacao.evaluations_results = historico["results"].tolist()
                    avaliacao.evaluations_length = historico["ep_lengths"].tolist()

        try:
            modelo.learn(
                total_timesteps=passos_adicionais,
                reset_num_timesteps=False,
                callback=[checkpoints, avaliacao, teclado],
            )
        except KeyboardInterrupt:
            print("\nInterrupção recebida. Salvando...")
        finally:
            teclado.restaurar_terminal()

        # Este arquivo é o salvamento normal. Checkpoints cobrem interrupções
        # abruptas que ocorram entre dois salvamentos periódicos.
        modelo.save(ARQUIVO_MODELO)
        print(f"Modelo salvo em: {ARQUIVO_MODELO}")
        print(f"Passos acumulados: {modelo.num_timesteps:,}")
    finally:
        if env_avaliacao is not None:
            env_avaliacao.close()
        env.close()


if __name__ == "__main__":
    main()
