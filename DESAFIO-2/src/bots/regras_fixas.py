"""Bot de regras fixas (parceiro e adversários do agente).

As regras nunca mudam, então para o agente as decisões dos bots fazem parte
do ambiente (ambiente estacionário, problema de agente único).

  Trucar/aumentar: tem manilha na mão e a dupla ganhou a 1ª rodada.
  Aceitar:         a dupla tem manilha ou um 3 (na mão ou na mesa da rodada).
  Correr:          nos outros casos.
  Jogar:           mata com a menor carta que vence a mesa; se não dá para
                   vencer (ou se está abrindo a rodada), joga a menor.
  Mão de 11:       a dupla joga se tem uma manilha ou dois 3; senão corre.

O bot só olha o que um jogador de verdade poderia ver: as próprias cartas e
o que está aberto na mesa (na mão de 11, também as cartas do parceiro).
"""
from truco.cartas import forca, FORCA_TRES, FORCA_MENOR_MANILHA


class BotRegras:
    nome = "regras"

    def quer_trucar(self, mao, rodadas, dupla):
        """Pede truco (ou aumenta) na própria vez, antes de jogar a carta."""
        tem_manilha = any(forca(c) >= FORCA_MENOR_MANILHA for c in mao)
        ganhou_primeira = len(rodadas) >= 1 and rodadas[0] == dupla
        return tem_manilha and ganhou_primeira

    def aceita(self, cartas_da_dupla):
        """Responde a um pedido: True aceita, False corre.

        `cartas_da_dupla`: cartas que a dupla ainda tem na mão mais as que
        ela já colocou na mesa na rodada atual.
        """
        return any(forca(c) >= FORCA_TRES for c in cartas_da_dupla)

    def joga_mao_de_11(self, cartas_da_dupla):
        """Na mão de 11 da dupla: True joga (vale 3), False corre (perde 1)."""
        forcas = [forca(c) for c in cartas_da_dupla]
        manilhas = sum(f >= FORCA_MENOR_MANILHA for f in forcas)
        treses = sum(f == FORCA_TRES for f in forcas)
        return manilhas >= 1 or treses >= 2

    def escolher_carta(self, mao, maior_da_mesa):
        """Devolve a carta a jogar. `maior_da_mesa` é 0 se a mesa está vazia."""
        ordenada = sorted(mao, key=forca)
        if maior_da_mesa > 0:
            for carta in ordenada:
                if forca(carta) > maior_da_mesa:
                    return carta          # mata barato
        return ordenada[0]                # abre fraco / não dá para vencer
