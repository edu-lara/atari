"""Criação padronizada dos ambientes Atari usados nos treinos."""

import ale_py
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)


def criar_ambiente_atari(
    nome,
    *,
    n_envs=1,
    seed=123,
    terminal_on_life_loss=False,
    clip_reward=True,
    subprocess=False,
):
    """Cria ambiente Atari com quatro frames e imagem em canais primeiro."""
    gym.register_envs(ale_py)
    kwargs = {
        "n_envs": n_envs,
        "seed": seed,
        "wrapper_kwargs": {
            "terminal_on_life_loss": terminal_on_life_loss,
            "clip_reward": clip_reward,
        },
    }
    if subprocess:
        kwargs["vec_env_cls"] = SubprocVecEnv
        kwargs["vec_env_kwargs"] = {"start_method": "spawn"}

    env = make_atari_env(nome, **kwargs)
    env = VecFrameStack(env, n_stack=4)
    return VecTransposeImage(env)
