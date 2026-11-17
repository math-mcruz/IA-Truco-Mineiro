"""Regras da casa que não dependem de quem está jogando.

Duplas: 0 = agente (assento 0) + parceiro (assento 2);
        1 = adversários (assentos 1 e 3).
"""

EMPATE = -1          # resultado de uma rodada empatada ("cangou")
TENTOS_PARTIDA = 12

# Escada das apostas: 1 -> truco (3) -> 6 -> 9 -> 12.
PROXIMO_VALOR = {1: 3, 3: 6, 6: 9, 9: 12}
VALORES = (1, 3, 6, 9, 12)

VALOR_MAO_DE_11 = 3      # mão de 11 jogada vale 3
CUSTO_CORRER_MAO_DE_11 = 1
VALOR_MAO_DE_FERRO = 1   # 11 a 11: quem vence a mão vence a partida


def dupla_de(jogador):
    return jogador % 2


def vencedor_da_rodada(mesa):
    """Quem venceu a rodada.

    `mesa` é a lista de jogadas na ordem em que aconteceram:
    (jogador, força), com força 0 para carta coberta.

    Devolve (dupla, jogador): a dupla vencedora (ou EMPATE) e o jogador que
    jogou a carta mais forte primeiro (None no empate). Se a carta mais forte
    aparece nas duas duplas, a rodada empata; se aparece duas vezes na mesma
    dupla, a dupla vence normalmente.
    """
    maior = max(f for _, f in mesa)
    donos = [j for j, f in mesa if f == maior]
    duplas = {dupla_de(j) for j in donos}
    if maior == 0 or len(duplas) > 1:
        return EMPATE, None
    return dupla_de(donos[0]), donos[0]


def vencedor_da_mao(rodadas):
    """Decide a mão a partir dos resultados das rodadas já jogadas.

    Devolve a dupla vencedora (0 ou 1), EMPATE se as três rodadas empataram
    (ninguém pontua) ou None se a mão ainda não acabou.

    Regras de empate adotadas:
      - 1ª empata            -> vence quem ganhar a 2ª (ou a 3ª, se a 2ª
                                também empatar);
      - 2ª ou 3ª empata      -> vence quem ganhou a 1ª;
      - as três empatam      -> ninguém pontua.
    """
    n = len(rodadas)
    if n < 2:
        return None
    r1, r2 = rodadas[0], rodadas[1]
    if n == 2:
        if r1 == EMPATE:
            return None if r2 == EMPATE else r2
        if r2 == EMPATE or r2 == r1:
            return r1
        return None          # 1 a 1: decide na 3ª
    r3 = rodadas[2]
    if r3 != EMPATE:
        return r3
    return r1                # r1 pode ser EMPATE: três empates
