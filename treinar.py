import os
import re
import select
import sys
from numbers import Real
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.logger import HumanOutputFormat
from stable_baselines3.common.utils import FloatSchedule

from ambientes_atari import criar_ambiente_atari
from jogos import escolher_jogo
from terminal import aguardar_continuar, limpar_terminal

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


NUM_AMBIENTES = 4
N_STEPS = 128
BATCH_SIZE = 256
N_EPOCHS = 4
TAXA_APRENDIZADO = 2.5e-4
CLIP_RANGE = 0.1
ENT_COEF = 0.01
INTERVALO_CHECKPOINT = 100_000
INTERVALO_AVALIACAO = 250_000
PARTIDAS_POR_AVALIACAO = 20


def perguntar_passos():
    while True:
        try:
            resposta = input(                
                "\nQuantos passos adicionais deseja executar?"
                "\n(padrão: 250.000): "
            ).strip()
            passos = 250_000 if not resposta else int(resposta)
        except ValueError:
            print("\nDigite um número inteiro positivo.")
            continue
        if passos == 0:
            limpar_terminal()
            return None
        if passos > 0:
            return passos
        print("O número de passos deve ser maior que zero.")


def aplicar_parametros_atari(modelo):
    modelo.n_steps = N_STEPS
    modelo.batch_size = BATCH_SIZE
    modelo.n_epochs = N_EPOCHS
    modelo.learning_rate = TAXA_APRENDIZADO
    modelo.lr_schedule = FloatSchedule(TAXA_APRENDIZADO)
    modelo.clip_range = FloatSchedule(CLIP_RANGE)
    modelo.ent_coef = ENT_COEF


def limpar_checkpoints_antigos(pasta_modelos, slug, manter=2):
    padrao = re.compile(rf"{re.escape(slug)}_(\d+)_steps\.zip$")
    checkpoints = []
    for arquivo in pasta_modelos.glob(f"{slug}_*_steps.zip"):
        correspondencia = padrao.fullmatch(arquivo.name)
        if correspondencia:
            checkpoints.append((int(correspondencia.group(1)), arquivo))

    # O sufixo numérico representa os passos acumulados; preserva os maiores.
    checkpoints.sort(key=lambda item: item[0], reverse=True)
    removidos = 0
    for _, arquivo in checkpoints[manter:]:
        arquivo.unlink()
        removidos += 1
    print(
        f"Checkpoints mantidos de {slug}: "
        f"{min(len(checkpoints), manter)} mais recentes; removidos: {removidos}."
    )


class AvaliacaoPorPassos(EvalCallback):
    def __init__(self, eval_env, intervalo_passos, **kwargs):
        self.intervalo_passos = intervalo_passos
        self.proxima_avaliacao = None
        super().__init__(eval_env, eval_freq=1, **kwargs)

    def _on_training_start(self):
        super()._on_training_start()
        self.proxima_avaliacao = (
            self.model.num_timesteps // self.intervalo_passos + 1
        ) * self.intervalo_passos

    def _on_step(self):
        if self.num_timesteps < self.proxima_avaliacao:
            return True
        while self.proxima_avaliacao <= self.num_timesteps:
            self.proxima_avaliacao += self.intervalo_passos
        return super()._on_step()

    def _on_training_end(self):
        # Garante que sessões menores que o intervalo também produzam uma
        # avaliação atualizada e possam salvar um novo melhor modelo.
        if (self.evaluations_timesteps
                and self.evaluations_timesteps[-1] == self.num_timesteps):
            return
        self.proxima_avaliacao = self.num_timesteps
        super()._on_step()


class PararComQ(BaseCallback):
    def __init__(self, passos_iniciais, passos_solicitados):
        super().__init__()
        self.fd = None
        self.configuracao_original = None
        self.passos_iniciais = passos_iniciais
        self.passos_solicitados = passos_solicitados

    def _on_training_start(self):
        # O SB3 configura o logger ao iniciar learn(); nesse ponto ele já
        # existe tanto para modelos novos quanto para modelos carregados.
        for indice, formato in enumerate(self.model.logger.output_formats):
            if isinstance(formato, HumanOutputFormat):
                self.model.logger.output_formats[indice] = HumanOutputFormatComAviso(
                    sys.stdout,
                    self.passos_iniciais,
                    self.passos_solicitados,
                )
                break

        if not sys.stdin.isatty():
            print("Entrada sem terminal interativo. Use Control + C para parar.")
        elif os.name == "nt" and msvcrt is not None:
            pass
        elif termios is not None and tty is not None:
            try:
                self.fd = sys.stdin.fileno()
                self.configuracao_original = termios.tcgetattr(self.fd)
                tty.setcbreak(self.fd)
            except (OSError, termios.error):
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
        if self.configuracao_original is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.configuracao_original)
            self.configuracao_original = None


class HumanOutputFormatComAviso(HumanOutputFormat):
    """Exibe a instrução de parada logo depois de cada tabela do PPO."""

    def __init__(self, filename_or_file, passos_iniciais, passos_solicitados):
        super().__init__(filename_or_file)
        self.passos_iniciais = passos_iniciais
        self.passos_solicitados = passos_solicitados

    @staticmethod
    def formatar_numero(valor):
        if not isinstance(valor, Real) or isinstance(valor, bool) or abs(valor) < 1000:
            return valor
        if float(valor).is_integer():
            return f"{int(valor):,}".replace(",", ".")
        formatado = f"{valor:,.2f}"
        return formatado.replace(",", "_").replace(".", ",").replace("_", ".")

    @staticmethod
    def formatar_duracao(segundos):
        horas, resto = divmod(max(0, int(segundos)), 3600)
        minutos, segundos = divmod(resto, 60)
        return f"{horas:02d}:{minutos:02d}:{segundos:02d}"

    def write(self, key_values, key_excluded, step=0):
        limpar_terminal()
        valores = dict(key_values)
        exclusoes = dict(key_excluded)
        passos_atuais = valores.get("time/total_timesteps", self.passos_iniciais)
        percentual = min(
            100.0,
            max(0.0, (passos_atuais - self.passos_iniciais)
                / self.passos_solicitados * 100),
        )
        passos_sessao = passos_atuais - self.passos_iniciais
        valores["time/meta_total"] = self.passos_iniciais + self.passos_solicitados
        exclusoes["time/meta_total"] = ()
        fps = valores.get("time/fps", 0)
        if isinstance(fps, Real) and fps > 0:
            passos_restantes = max(0, self.passos_solicitados - passos_sessao)
            estimativa_restante = self.formatar_duracao(
                passos_restantes / fps
            )
        else:
            estimativa_restante = "calculando"
        if "time/time_elapsed" in valores:
            valores["time/time_elapsed"] = self.formatar_duracao(
                valores["time/time_elapsed"]
            )
        valores_exibicao = {
            chave: self.formatar_numero(valor)
            for chave, valor in valores.items()
        }
        super().write(valores_exibicao, exclusoes, step)
        largura_barra = 20
        preenchidos = round(percentual / 100 * largura_barra)
        barra = "#" * preenchidos + "-" * (largura_barra - preenchidos)
        print(
            f"{self.formatar_numero(passos_sessao)} / "
            f"{self.formatar_numero(self.passos_solicitados)} passos",
            flush=True,
        )
        print(
            f"Sessão [{barra}] {percentual:.0f}%",
            flush=True,
        )
        print(f"Tempo estimado restante: ~{estimativa_restante}", flush=True)
        print("Pressione Q neste terminal para parar e salvar.", flush=True)


def treinar(nome, ambiente_id, slug, passos_adicionais):
    pasta_modelos = Path("modelos")
    arquivo_modelo = pasta_modelos / f"{slug}_atual.zip"
    pasta_melhor = pasta_modelos / f"{slug}_melhor"
    prefixo_checkpoint = slug
    padrao_checkpoint = re.compile(rf"{re.escape(slug)}_(\d+)_steps\.zip$")
    checkpoints_existentes = []
    for arquivo in pasta_modelos.glob(f"{slug}_*_steps.zip"):
        correspondencia = padrao_checkpoint.fullmatch(arquivo.name)
        if correspondencia:
            checkpoints_existentes.append((int(correspondencia.group(1)), arquivo))
    checkpoint = max(checkpoints_existentes, default=(0, None), key=lambda item: item[0])[1]

    pasta_modelos.mkdir(parents=True, exist_ok=True)
    env = criar_ambiente_atari(
        ambiente_id, n_envs=NUM_AMBIENTES, seed=42,
        terminal_on_life_loss=False, subprocess=True,
        iniciar_com_fire=(slug == "video_pinball"),
    )
    env_avaliacao = None
    try:
        env_avaliacao = criar_ambiente_atari(
            ambiente_id, n_envs=1, seed=123, terminal_on_life_loss=False,
            clip_reward=(slug != "breakout"),
            iniciar_com_fire=(slug == "video_pinball"),
        )
        if checkpoint is not None and (
            not arquivo_modelo.exists()
            or checkpoint.stat().st_mtime > arquivo_modelo.stat().st_mtime
        ):
            arquivo_para_carregar = checkpoint
            print(f"Retomando do checkpoint: {checkpoint}")
        elif arquivo_modelo.exists():
            arquivo_para_carregar = arquivo_modelo
            print("Modelo salvo carregado. Continuando o aprendizado.")
        else:
            arquivo_para_carregar = checkpoint
            if checkpoint is not None:
                print(f"Retomando do checkpoint: {checkpoint}")

        if arquivo_para_carregar is not None:
            modelo = PPO.load(arquivo_para_carregar, env=env, device="cpu")
        else:
            modelo = PPO(
                "CnnPolicy", env, device="cpu", verbose=1, seed=42,
                n_steps=N_STEPS, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS,
                learning_rate=TAXA_APRENDIZADO, clip_range=CLIP_RANGE,
                ent_coef=ENT_COEF, tensorboard_log=f"./logs/{slug}/",
            )
            print(f"Criando um modelo novo para {nome}.")
        aplicar_parametros_atari(modelo)
        inicio = modelo.num_timesteps
        print(f"Passos acumulados: {inicio:,}")
        print(f"Passos adicionais solicitados: {passos_adicionais:,}")
        print(f"Total previsto: {inicio + passos_adicionais:,}")
        print("O PPO pode passar um pouco do total solicitado para concluir um ciclo.")

        teclado = PararComQ(inicio, passos_adicionais)
        callback_checkpoint = CheckpointCallback(
            save_freq=max(INTERVALO_CHECKPOINT // NUM_AMBIENTES, 1),
            save_path=str(pasta_modelos), name_prefix=prefixo_checkpoint,
        )
        pasta_melhor.mkdir(parents=True, exist_ok=True)
        pasta_log_avaliacao = Path(f"logs/{slug}_avaliacao")
        arquivo_historico = pasta_log_avaliacao / "evaluations.npz"
        partidas_por_avaliacao = 5 if slug == "video_pinball" else PARTIDAS_POR_AVALIACAO
        avaliacao = AvaliacaoPorPassos(
            env_avaliacao,
            intervalo_passos=(100_000 if slug == "video_pinball" else INTERVALO_AVALIACAO),
            best_model_save_path=str(pasta_melhor), log_path=str(pasta_log_avaliacao),
            n_eval_episodes=partidas_por_avaliacao, deterministic=True, verbose=1,
        )
        arquivo_melhor = pasta_melhor / "best_model.zip"
        if arquivo_historico.exists() and arquivo_melhor.exists():
            with np.load(arquivo_historico) as historico:
                medias = historico["results"].mean(axis=1)
                if len(medias):
                    avaliacao.best_mean_reward = float(medias.max())
                    avaliacao.evaluations_timesteps = historico["timesteps"].tolist()
                    avaliacao.evaluations_results = historico["results"].tolist()
                    avaliacao.evaluations_length = historico["ep_lengths"].tolist()
        try:
            modelo.learn(
                total_timesteps=passos_adicionais, reset_num_timesteps=False,
                callback=[callback_checkpoint, avaliacao, teclado],
            )
        except KeyboardInterrupt:
            print("\nInterrupção recebida. Salvando...")
        finally:
            teclado.restaurar_terminal()
        print()
        modelo.save(arquivo_modelo)
        print(f"Modelo salvo em: {arquivo_modelo}")
        print(f"Passos acumulados: {modelo.num_timesteps:,}")
        limpar_checkpoints_antigos(pasta_modelos, slug)
    finally:
        if env_avaliacao is not None:
            env_avaliacao.close()
        env.close()
    aguardar_continuar()


def main():
    limpar_terminal()
    jogo = escolher_jogo("treinar")
    if jogo is None:
        return
    nome, ambiente_id, slug = jogo
    passos = perguntar_passos()
    if passos is None:
        return
    treinar(nome, ambiente_id, slug, passos)


if __name__ == "__main__":
    main()
