import numpy as np
import pytest

from auxiliares.politicas import PoliticaAleatoria, PoliticaRegras
from truco.ambiente import (ABRIR_FORTE, ABRIR_FRACO, ACEITAR, BLOCOS_OBS, CORRER,
                            DEIXAR_FAZER, ESCONDER, MATAR_BARATO, MATAR_FORTE,
                            N_ACOES, TAM_OBS, TRUCAR, TrucoEnv)
from truco.cartas import Carta


def mao(*textos):
    return [Carta(t[:-1], t[-1]) for t in textos]


def validas(info):
    return set(np.flatnonzero(info["mascara"]))


def bloco(obs, nome):
    i = 0
    for n, tamanho in BLOCOS_OBS:
        if n == nome:
            return obs[i:i + tamanho]
        i += tamanho
    raise KeyError(nome)


FRACAS_1 = mao("Q♣", "J♣", "K♣")
FRACAS_2 = mao("Q♦", "J♦", "K♦")
FRACAS_3 = mao("Q♠", "J♠", "K♠")


def novo(cartas, placar=(0, 0), primeiro=0):
    env = TrucoEnv()
    obs, info = env.reset(seed=0, options={
        "placar": placar, "primeiro": primeiro, "cartas": cartas})
    return env, obs, info


# ------------------------------------------------------------- invariantes
def test_partidas_aleatorias_respeitam_as_invariantes():
    env = TrucoEnv()
    politica = PoliticaAleatoria(seed=1)
    for partida in range(300):
        obs, info = env.reset(seed=partida)
        retorno, terminou = 0.0, False
        while not terminou:
            assert obs.shape == (TAM_OBS,) and obs.dtype == np.float32
            assert obs.min() >= 0.0 and obs.max() <= 1.0
            assert info["mascara"].any()
            acao = politica.agir(env, obs, info["mascara"])
            obs, r, terminou, truncou, info = env.step(acao)
            assert not truncou
            retorno += r
        # O retorno do episódio é o saldo de tentos da partida.
        assert retorno == env.placar[0] - env.placar[1]
        assert max(env.placar) >= 12 and min(env.placar) < 12
        assert info["venceu"] == (env.placar[0] >= 12)


def test_acao_invalida_e_recusada():
    env, _, info = novo([[mao("4♣", "K♥", "Q♥"), FRACAS_1, FRACAS_2, FRACAS_3]])
    assert not info["mascara"][CORRER]
    with pytest.raises(ValueError):
        env.step(CORRER)


def test_quatro_bots_iguais_ficam_perto_de_50_por_cento():
    env, politica, vitorias, n = TrucoEnv(), PoliticaRegras(), 0, 2000
    for partida in range(n):
        obs, info = env.reset(seed=partida)
        terminou = False
        while not terminou:
            obs, _, terminou, _, info = env.step(politica.agir(env, obs, info["mascara"]))
        vitorias += info["venceu"]
    assert 0.45 < vitorias / n < 0.55


# ----------------------------------------------------------------- apostas
def test_abrindo_a_rodada_as_acoes_sao_abrir_ou_trucar():
    _, _, info = novo([[mao("4♣", "K♥", "Q♥"), FRACAS_1, FRACAS_2, FRACAS_3]])
    assert info["decisao"] == "carta"
    assert validas(info) == {ABRIR_FORTE, ABRIR_FRACO, TRUCAR}


def test_trucar_e_os_adversarios_correm_vale_1():
    env, _, _ = novo([[mao("4♣", "K♥", "Q♥"), FRACAS_1, FRACAS_2, FRACAS_3]])
    _, r, terminou, _, info = env.step(TRUCAR)      # eles não têm manilha nem 3
    assert r == 1 and not terminou
    assert info["placar"] == (1, 0)


def test_truco_aceito_vale_3_e_gera_novo_estado_para_a_carta():
    adv = mao("3♣", "J♣", "K♣")                     # tem um 3: aceita
    env, _, _ = novo([[mao("4♣", "7♥", "Q♥"), adv, FRACAS_2, FRACAS_3]])
    obs, r, _, _, info = env.step(TRUCAR)
    assert r == 0 and info["decisao"] == "carta"
    assert validas(info) == {ABRIR_FORTE, ABRIR_FRACO}   # o aumento agora é deles
    assert bloco(obs, "valor_da_mao").tolist() == [0, 1, 0, 0, 0]
    _, r, _, _, _ = env.step(ABRIR_FORTE)           # Zap faz a 1ª
    assert r == 0
    _, r, _, _, info = env.step(ABRIR_FORTE)        # 7♥ faz a 2ª
    assert r == 3 and info["placar"] == (3, 0)


def cenario_truco_dos_adversarios():
    """Adversários fazem a 1ª; na 2ª o da direita (com o 7♦) pede truco."""
    cartas = [[mao("Q♥", "J♥", "K♥"), mao("7♦", "Q♣", "J♣"),
               FRACAS_2, mao("3♠", "J♠", "K♠")]]
    env, _, info = novo(cartas, primeiro=1)
    # 1ª rodada: Q♣, J♦, K♠ na mesa; nenhuma carta minha vence o K.
    assert validas(info) == {DEIXAR_FAZER, TRUCAR}        # sem coberta na 1ª
    _, r, _, _, info = env.step(DEIXAR_FAZER)
    assert r == 0
    # 2ª rodada: o da esquerda abre com J♠ e agora posso jogar coberta.
    assert validas(info) == {MATAR_BARATO, MATAR_FORTE, DEIXAR_FAZER,
                             ESCONDER, TRUCAR}
    obs, r, _, _, info = env.step(MATAR_BARATO)
    assert r == 0 and info["decisao"] == "aposta"
    assert validas(info) == {TRUCAR, ACEITAR, CORRER}
    assert bloco(obs, "truco_pendente")[0] == 1
    assert bloco(obs, "valor_pedido").tolist() == [1, 0, 0, 0]
    assert bloco(obs, "quem_pediu").tolist() == [1, 0]
    return env


def test_correr_do_truco_perde_1():
    env = cenario_truco_dos_adversarios()
    _, r, _, _, info = env.step(CORRER)
    assert r == -1 and info["placar"] == (0, 1)


def test_aceitar_o_truco_e_perder_custa_3():
    env = cenario_truco_dos_adversarios()
    _, r, _, _, info = env.step(ACEITAR)            # o 7♦ faz a 2ª para eles
    assert r == -3 and info["placar"] == (0, 3)


def test_responder_com_aumento_leva_a_mao_para_6():
    env = cenario_truco_dos_adversarios()
    _, r, _, _, info = env.step(TRUCAR)             # "seis!"; eles aceitam
    assert r == -6 and info["placar"] == (0, 6)


def test_carta_coberta_sai_da_mao_e_nao_disputa():
    cartas = [[mao("Q♥", "J♥", "K♥"), mao("7♦", "Q♣", "J♣"),
               FRACAS_2, mao("3♠", "J♠", "K♠")]]
    env, _, _ = novo(cartas, primeiro=1)
    env.step(DEIXAR_FAZER)
    env.step(ESCONDER)                              # J♥ virado para baixo
    jogador, carta, coberta = env.mao.mesa[1]
    assert (jogador, carta, coberta) == (0, Carta("J", "♥"), True)
    assert env.mao.cartas[0] == mao("K♥")


# --------------------------------------------------------------- mão de 11
def test_mao_de_11_so_aceitar_ou_correr_e_mostra_o_parceiro():
    cartas = [[mao("4♣", "7♥", "3♥"), FRACAS_1, mao("3♦", "2♦", "K♦"), FRACAS_3]]
    _, obs, info = novo(cartas, placar=(11, 5))
    assert info["decisao"] == "onze"
    assert validas(info) == {ACEITAR, CORRER}
    assert bloco(obs, "mao_de_11").tolist() == [1, 0, 1]
    assert bloco(obs, "cartas_do_parceiro").sum() == pytest.approx(1.0)


def test_correr_na_mao_de_11_perde_1():
    cartas = [[mao("Q♥", "J♥", "K♥"), FRACAS_1, FRACAS_2, FRACAS_3]]
    env, _, _ = novo(cartas, placar=(11, 5))
    _, r, terminou, _, info = env.step(CORRER)
    assert r == -1 and not terminou and info["placar"] == (11, 6)
    assert info["decisao"] == "onze"                # continua com 11


def test_mao_de_11_jogada_vale_3_sem_truco_e_fecha_a_partida():
    cartas = [[mao("4♣", "7♥", "3♥"), FRACAS_1, mao("3♦", "2♦", "K♦"), FRACAS_3]]
    env, _, _ = novo(cartas, placar=(11, 5))
    obs, r, _, _, info = env.step(ACEITAR)
    assert r == 0 and validas(info) == {ABRIR_FORTE, ABRIR_FRACO}   # sem Trucar
    assert bloco(obs, "valor_da_mao").tolist() == [0, 1, 0, 0, 0]
    env.step(ABRIR_FORTE)
    _, r, terminou, _, info = env.step(ABRIR_FORTE)
    assert r == 3 and terminou and info["venceu"]


def test_mao_de_11_dos_adversarios_sem_jogo_eles_correm():
    cartas = [[mao("Q♥", "J♥", "K♥"), FRACAS_1, FRACAS_2, FRACAS_3]]
    env, _, _ = novo(cartas, placar=(5, 11))
    assert env.placar == [6, 11]


def test_mao_de_ferro_e_resolvida_pelo_ambiente():
    fortes_1, fortes_3 = mao("4♣", "7♥", "3♣"), mao("A♠", "7♦", "3♠")
    cartas = [
        [mao("Q♥", "J♥", "K♥"), fortes_1, FRACAS_2, fortes_3],     # mão de 11
        [mao("4♣", "7♥", "A♠"), FRACAS_1, FRACAS_2, FRACAS_3],     # mão de ferro
    ]
    env, _, _ = novo(cartas, placar=(11, 8))
    env.step(ACEITAR)
    _, r, _, _, info = env.step(ABRIR_FRACO)
    assert r == 0
    assert validas(info) == {DEIXAR_FAZER, ESCONDER}
    # Perdemos a mão de 11 (−3): 11 a 11. A mão de ferro é jogada às cegas
    # dentro do mesmo passo e nós vencemos (+1).
    _, r, terminou, _, info = env.step(DEIXAR_FAZER)
    assert r == -2 and terminou and info["venceu"]
    assert env.placar == [12, 11]


def test_observacao_tem_o_tamanho_dos_blocos():
    assert TAM_OBS == sum(t for _, t in BLOCOS_OBS)
    env = TrucoEnv()
    obs, info = env.reset(seed=3)
    assert obs.shape == env.observation_space.shape == (TAM_OBS,)
    assert info["mascara"].shape == (N_ACOES,)
    assert bloco(obs, "mascara").astype(bool).tolist() == info["mascara"].tolist()
