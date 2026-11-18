import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from ambientes_atari import criar_ambiente_atari
from assistir import escolher_modelo
from jogos import escolher_jogo
from terminal import aguardar_continuar, limpar_terminal


NUM_PARTIDAS = 10
NUM_PARTIDAS_VIDEO_PINBALL = 5


def partidas_padrao(slug):
    return NUM_PARTIDAS_VIDEO_PINBALL if slug == "video_pinball" else NUM_PARTIDAS


def criar_callback_progresso(rotulo, total):
    concluidas = 0

    def mostrar(locals_, _globals):
        nonlocal concluidas
        if locals_["done"]:
            concluidas += 1
            print(f"  {rotulo}: partida {concluidas}/{total} concluída.", flush=True)

    return mostrar


class AgenteAleatorio:
    def __init__(self, espaco_acoes):
        self.espaco_acoes = espaco_acoes
        self.espaco_acoes.seed(123)

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        return np.array([self.espaco_acoes.sample()]), state


def criar_ambiente(ambiente_id, slug):
    return criar_ambiente_atari(
        ambiente_id,
        n_envs=1,
        seed=123,
        terminal_on_life_loss=False,
        clip_reward=(slug != "breakout"),
    )


def avaliar(nome, ambiente_id, slug, num_partidas, modelo=None):
    env = criar_ambiente(ambiente_id, slug)
    try:
        agente = modelo if modelo is not None else AgenteAleatorio(env.action_space)
        print(f"\nAvaliando {nome}: {num_partidas} partidas...", flush=True)
        recompensas, _ = evaluate_policy(
            agente,
            env,
            n_eval_episodes=num_partidas,
            deterministic=True,
            return_episode_rewards=True,
            callback=criar_callback_progresso(nome, num_partidas),
        )
        media = float(np.mean(recompensas))
        desvio = float(np.std(recompensas))
        positivas = sum(recompensa > 0 for recompensa in recompensas)
        print(f"Resultados: {recompensas}")
        print(f"Recompensa média: {media:.2f}")
        print(f"Desvio padrão: {desvio:.2f}")
        print(f"Partidas com recompensa positiva: {positivas}/{num_partidas}")
        return media
    finally:
        env.close()


def main():
    limpar_terminal()
    while True:
        jogo = escolher_jogo("avaliar")
        if jogo is None:
            return
        nome, ambiente_id, slug = jogo
        modelo_selecionado = escolher_modelo(nome, slug, "avaliar")
        if modelo_selecionado is not None:
            arquivo_modelo, modelo = modelo_selecionado
            break

    num_partidas = partidas_padrao(slug)
    print(f"Avaliando modelo: {arquivo_modelo}")
    passos = f"{modelo.num_timesteps:,}".replace(",", ".")
    print(f"Passos acumulados no modelo: {passos}")
    media_aleatorio = avaliar(
        "agente aleatório", ambiente_id, slug, num_partidas
    )
    media_treinado = avaliar(
        "agente treinado", ambiente_id, slug, num_partidas, modelo
    )
    print("\nCOMPARAÇÃO FINAL")
    print(f"Agente aleatório: {media_aleatorio:.2f}")
    print(f"Agente treinado: {media_treinado:.2f}")
    print(f"Diferença: {media_treinado - media_aleatorio:+.2f}")
    aguardar_continuar()


if __name__ == "__main__":
    main()
