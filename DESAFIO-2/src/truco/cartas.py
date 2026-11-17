"""Baralho vazio do truco mineiro (27 cartas) e a força de cada carta.

Da mais forte para a mais fraca:
  manilhas fixas: Zap (4♣) > 7♥ > Espadilha (A♠) > 7♦
  comuns:         3 > 2 > A > K > J > Q   (naipe não desempata)
"""
from collections import namedtuple

Carta = namedtuple("Carta", ["valor", "naipe"])

NAIPES = ("♣", "♥", "♠", "♦")

# Força das cartas comuns (o naipe não importa).
FORCA_COMUM = {"Q": 1, "J": 2, "K": 3, "A": 4, "2": 5, "3": 6}

# Manilhas fixas: valem pela carta exata (valor + naipe).
MANILHAS = {
    Carta("7", "♦"): 7,
    Carta("A", "♠"): 8,   # Espadilha
    Carta("7", "♥"): 9,
    Carta("4", "♣"): 10,  # Zap
}

FORCA_MAX = 10
FORCA_TRES = FORCA_COMUM["3"]      # 6
FORCA_MENOR_MANILHA = 7            # 7♦


def forca(carta):
    """Força da carta, de 1 (Q) a 10 (Zap)."""
    if carta in MANILHAS:
        return MANILHAS[carta]
    return FORCA_COMUM[carta.valor]


def eh_manilha(carta):
    return carta in MANILHAS


def baralho():
    """As 27 cartas: 20 comuns (Q, J, K, 2, 3 nos quatro naipes),
    3 ases comuns (copas, ouros e paus) e as 4 manilhas."""
    cartas = [Carta(v, n) for v in ("Q", "J", "K", "2", "3") for n in NAIPES]
    cartas += [Carta("A", n) for n in ("♥", "♦", "♣")]
    cartas += list(MANILHAS)
    return cartas


def nome(carta):
    return f"{carta.valor}{carta.naipe}"
