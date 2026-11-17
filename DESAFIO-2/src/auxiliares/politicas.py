"""Baselines que jogam no assento do agente, usando as mesmas 9 ações.

Servem de régua para o DQN:
  - PoliticaAleatoria: sorteia uma ação válida (respeita a máscara).
  - PoliticaRegras: as regras do bot escritas com as ações do agente. Com os
    quatro jogadores iguais, a taxa de vitória esperada é 50%.
"""
import numpy as np

from bots.regras_fixas import BotRegras
from truco.ambiente import (ABRIR_FRACO, ACEITAR, AGENTE, CORRER, DECISAO_APOSTA,
                            DECISAO_ONZE, DEIXAR_FAZER, MATAR_BARATO, PARCEIRO,
                            TRUCAR)
from truco.cartas import forca


class PoliticaAleatoria:
    nome = "Aleatório"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def agir(self, env, obs, mascara):
        return int(self.rng.choice(np.flatnonzero(mascara)))


class PoliticaRegras:
    """O bot de regras sentado no lugar do agente.

    Para ser igual aos outros bots, ele responde às apostas olhando as cartas
    da dupla (as dele e as do parceiro), informação que o DQN não tem.
    """
    nome = "Bot de regras"

    def __init__(self):
        self.bot = BotRegras()

    def agir(self, env, obs, mascara):
        m = env.mao
        minhas = m.cartas[AGENTE]
        if env._decisao == DECISAO_ONZE:
            joga = self.bot.joga_mao_de_11(minhas + m.cartas[PARCEIRO])
            return ACEITAR if joga else CORRER
        if env._decisao == DECISAO_APOSTA:
            vivas = minhas + m.cartas[PARCEIRO] + [
                c for dono, c, _ in m.mesa if dono % 2 == 0]
            return ACEITAR if self.bot.aceita(vivas) else CORRER
        if mascara[TRUCAR] and self.bot.quer_trucar(minhas, m.rodadas, 0):
            return TRUCAR
        if not m.mesa:
            return ABRIR_FRACO
        maior = env._maior_da_mesa()
        if any(forca(c) > maior for c in minhas):
            return MATAR_BARATO
        return DEIXAR_FAZER
