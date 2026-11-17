from truco.cartas import Carta, baralho, forca
from truco.regras import EMPATE, vencedor_da_mao, vencedor_da_rodada


def C(texto):
    return Carta(texto[:-1], texto[-1])


def test_baralho_vazio_tem_27_cartas_distintas():
    cartas = baralho()
    assert len(cartas) == 27
    assert len(set(cartas)) == 27


def test_ordem_de_forca():
    ordem = ["4♣", "7♥", "A♠", "7♦", "3♥", "2♥", "A♥", "K♥", "J♥", "Q♥"]
    forcas = [forca(C(t)) for t in ordem]
    assert forcas == sorted(forcas, reverse=True)
    assert len(set(forcas)) == 10


def test_naipe_nao_desempata_cartas_comuns():
    assert forca(C("3♥")) == forca(C("3♠")) == forca(C("3♣")) == forca(C("3♦"))
    assert forca(C("A♥")) < forca(C("A♠"))        # só a Espadilha é manilha


def test_rodada_vence_a_maior_carta():
    assert vencedor_da_rodada([(0, 3), (1, 6), (2, 10), (3, 1)]) == (0, 2)
    assert vencedor_da_rodada([(1, 6), (2, 5), (3, 2), (0, 1)]) == (1, 1)


def test_rodada_empata_quando_a_maior_esta_nas_duas_duplas():
    assert vencedor_da_rodada([(0, 6), (1, 6), (2, 1), (3, 2)]) == (EMPATE, None)


def test_mesma_carta_na_mesma_dupla_nao_empata():
    assert vencedor_da_rodada([(0, 6), (1, 2), (2, 6), (3, 1)]) == (0, 0)


def test_carta_coberta_perde_para_qualquer_carta():
    assert vencedor_da_rodada([(1, 1), (2, 2), (3, 1), (0, 0)]) == (0, 2)
    assert vencedor_da_rodada([(1, 1), (0, 0), (3, 1), (2, 0)]) == (1, 1)


def test_mao_ganha_quem_faz_duas_rodadas():
    assert vencedor_da_mao([0]) is None
    assert vencedor_da_mao([0, 0]) == 0
    assert vencedor_da_mao([1, 0]) is None
    assert vencedor_da_mao([1, 0, 0]) == 0
    assert vencedor_da_mao([0, 1, 1]) == 1


def test_primeira_empata_vence_quem_ganhar_a_segunda():
    assert vencedor_da_mao([EMPATE]) is None
    assert vencedor_da_mao([EMPATE, 1]) == 1
    assert vencedor_da_mao([EMPATE, EMPATE]) is None
    assert vencedor_da_mao([EMPATE, EMPATE, 0]) == 0


def test_segunda_ou_terceira_empata_vence_quem_ganhou_a_primeira():
    assert vencedor_da_mao([1, EMPATE]) == 1
    assert vencedor_da_mao([0, 1, EMPATE]) == 0


def test_tres_empates_ninguem_pontua():
    assert vencedor_da_mao([EMPATE, EMPATE, EMPATE]) == EMPATE
