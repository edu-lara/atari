from stable_baselines3.common.evaluation import evaluate_policy

from assistir import escolher_modelo
from avaliar import (
    criar_ambiente,
    criar_callback_progresso,
    partidas_padrao,
)
from jogos import escolher_jogo
from terminal import aguardar_continuar, limpar_terminal


def main():
    limpar_terminal()
    while True:
        jogo = escolher_jogo("diagnosticar")
        if jogo is None:
            return
        nome, ambiente_id, slug = jogo
        modelo_selecionado = escolher_modelo(nome, slug, "diagnosticar")
        if modelo_selecionado is not None:
            arquivo_modelo, modelo = modelo_selecionado
            break

    num_partidas = partidas_padrao(slug)
    env = criar_ambiente(ambiente_id, slug)
    try:
        print(f"Modelo: {arquivo_modelo}")
        passos = f"{modelo.num_timesteps:,}".replace(",", ".")
        print(f"Passos acumulados: {passos}")
        print(f"Avaliando {num_partidas} partidas em cada modo...", flush=True)
        media_det, desvio_det = evaluate_policy(
            modelo,
            env,
            n_eval_episodes=num_partidas,
            deterministic=True,
            callback=criar_callback_progresso("determinístico", num_partidas),
        )
        media_prob, desvio_prob = evaluate_policy(
            modelo,
            env,
            n_eval_episodes=num_partidas,
            deterministic=False,
            callback=criar_callback_progresso("probabilístico", num_partidas),
        )
        print("\nRESULTADO DO DIAGNÓSTICO")
        print(f"Modo determinístico: média {media_det:.2f}, desvio padrão {desvio_det:.2f}")
        print(f"Modo probabilístico: média {media_prob:.2f}, desvio padrão {desvio_prob:.2f}")
    finally:
        env.close()
    aguardar_continuar()


if __name__ == "__main__":
    main()
