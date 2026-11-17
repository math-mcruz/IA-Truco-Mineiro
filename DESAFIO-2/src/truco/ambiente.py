"""Ambiente de truco mineiro 2 contra 2 na interface do Gymnasium.

O agente controla o assento 0. O parceiro (assento 2) e os dois adversários
(assentos 1 e 3) são bots de regras fixas que rodam dentro do ambiente.

  obs, info = env.reset(seed=...)
  obs, recompensa, terminou, truncou, info = env.step(acao)

`info["mascara"]` diz quais das 9 ações são válidas no estado atual.

Um episódio é a partida inteira (até 12 tentos). Cada passo é uma decisão do
agente: jogar uma carta, responder a uma aposta ou decidir a mão de 11. A
recompensa é 0 durante a mão e, quando a mão acaba, vale os tentos ganhos
menos os tentos perdidos pela dupla (recompensa esparsa).

A partida é escrita como um gerador do Python: o código das regras corre em
linha reta e dá `yield` sempre que precisa de uma decisão do agente.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bots.regras_fixas import BotRegras
from truco import regras
from truco.cartas import baralho, forca, nome, FORCA_MAX
from truco.regras import EMPATE, dupla_de

# --- As 9 ações (estratégias, não cartas) ---------------------------------
(MATAR_BARATO, MATAR_FORTE, DEIXAR_FAZER, ESCONDER,
 ABRIR_FORTE, ABRIR_FRACO, TRUCAR, ACEITAR, CORRER) = range(9)
N_ACOES = 9
NOMES_ACOES = ["Matar barato", "Matar forte", "Deixar fazer", "Esconder",
               "Abrir forte", "Abrir fraco", "Trucar", "Aceitar", "Correr"]

# O jogo corre para a direita (sentido anti-horário): depois do agente (0)
# joga o adversário à direita dele (1), depois o parceiro (2) e por fim o
# adversário à esquerda (3).
AGENTE, ADV_DIREITA, PARCEIRO, ADV_ESQUERDA = 0, 1, 2, 3
NOMES_JOGADORES = ["Agente", "Adv. direita", "Parceiro", "Adv. esquerda"]
NOMES_DUPLAS = ["nós", "eles"]

# Tipos de decisão do agente.
DECISAO_CARTA, DECISAO_APOSTA, DECISAO_ONZE = "carta", "aposta", "onze"

# Blocos da observação, na ordem, com o tamanho de cada um.
BLOCOS_OBS = [
    ("mao_por_forca", 10),          # quantas cartas de cada força eu tenho
    ("mao_ordenada", 3),            # minhas cartas, da maior para a menor
    ("jogadas_eu", 10),             # cartas já jogadas nesta mão, por jogador
    ("jogadas_adv_direita", 10),
    ("jogadas_parceiro", 10),
    ("jogadas_adv_esquerda", 10),
    ("rodadas", 12),                # 3 rodadas x (nós, eles, empate, não jogada)
    ("maior_da_mesa", 1),
    ("dono_da_maior", 3),           # mesa vazia, nossa dupla, adversário
    ("cartas_na_mesa", 3),          # adv. direita, parceiro, adv. esquerda
    ("posicao_na_rodada", 4),       # quantos já jogaram antes de mim
    ("rodada_atual", 3),
    ("valor_da_mao", 5),            # 1, 3, 6, 9, 12
    ("truco_pendente", 1),
    ("valor_pedido", 4),            # 3, 6, 9, 12
    ("quem_pediu", 2),              # adv. direita, adv. esquerda
    ("direito_de_aumentar", 3),     # livre, nossa dupla, dupla deles
    ("ultimo_pedido_aceito", 5),    # ninguém, eu, adv. dir., parceiro, adv. esq.
    ("placar", 2),                  # nós, eles (dividido por 12)
    ("mao_de_11", 3),               # nossa, deles, estou decidindo agora
    ("cartas_do_parceiro", 10),     # só na mão de 11 da nossa dupla
    ("quem_e_mao", 4),              # quem abriu a mão
    ("mascara", N_ACOES),           # ações válidas (dá para deduzir do resto)
]
TAM_OBS = sum(tamanho for _, tamanho in BLOCOS_OBS)


class _Mao:
    """Tudo o que acontece dentro de uma mão."""

    def __init__(self, cartas, primeiro):
        self.cartas = cartas                      # o que cada jogador ainda tem
        self.jogadas = [[] for _ in range(4)]     # forças já jogadas por jogador
        self.rodadas = []                         # 0, 1 ou EMPATE por rodada
        self.mesa = []                            # rodada atual: (jogador, carta, coberta)
        self.valor = 1
        self.direito = None        # dupla que pode aumentar (None = qualquer uma)
        self.ultimo_pedido = None  # jogador que fez o último pedido aceito
        self.pendente = None       # (valor pedido, quem pediu) esperando o agente
        self.mao_de_11 = None      # dupla com 11 tentos, se for mão de 11
        self.decidindo_11 = False
        self.primeiro = primeiro   # quem é "mão" (abre a 1ª rodada)


class TrucoEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, bots=None, registrar=False):
        """`bots`: dicionário {assento: bot} para os assentos 1, 2 e 3.
        `registrar`: guarda em `self.eventos` o que acontece na mesa, em
        texto (usado pelo replay; não muda o jogo)."""
        self.registrar = registrar
        self.eventos = []
        padrao = BotRegras()
        self.bots = {1: padrao, 2: padrao, 3: padrao}
        if bots:
            self.bots.update(bots)
        self.observation_space = spaces.Box(0.0, 1.0, (TAM_OBS,), np.float32)
        self.action_space = spaces.Discrete(N_ACOES)
        self.mao = None
        self._fim = True

    # ------------------------------------------------------------------ API
    def reset(self, seed=None, options=None):
        """Começa uma partida nova.

        `options` serve para montar situações nos testes:
          "placar":   (nós, eles) no começo da partida;
          "primeiro": assento que é "mão" na 1ª mão;
          "cartas":   lista de distribuições, uma por mão, cada uma com as
                      3 cartas de cada assento (depois volta ao sorteio).
        """
        super().reset(seed=seed)
        options = options or {}
        self.placar = list(options.get("placar", (0, 0)))   # [nós, eles]
        self._primeiro_fixo = options.get("primeiro")
        self._cartas_fixas = list(options.get("cartas", []))
        self.n_maos = 0
        self._recompensa = 0.0
        self._decisao = None
        self._fim = False
        self._jogo = self._partida()
        self._avancar(None)
        return self._obs(), self._info()

    def step(self, acao):
        if self._fim:
            raise RuntimeError("A partida acabou; chame reset().")
        acao = int(acao)
        if not self.mascara()[acao]:
            raise ValueError(f"Ação inválida neste estado: {NOMES_ACOES[acao]}")
        self._avancar(acao)
        recompensa, self._recompensa = self._recompensa, 0.0
        return self._obs(), recompensa, self._fim, False, self._info()

    def _evento(self, texto):
        if self.registrar:
            self.eventos.append(texto)

    def mascara(self):
        """Vetor booleano com as ações permitidas no estado atual."""
        mask = np.zeros(N_ACOES, dtype=bool)
        if self._fim:
            mask[:] = True            # estado terminal: a máscara não é usada
            return mask
        m = self.mao
        if self._decisao == DECISAO_ONZE:          # mão de 11: jogar ou correr
            mask[[ACEITAR, CORRER]] = True
        elif self._decisao == DECISAO_APOSTA:      # responder ao pedido deles
            mask[[ACEITAR, CORRER]] = True
            mask[TRUCAR] = m.pendente[0] < 12      # não existe aumento acima de 12
        else:                                      # jogar uma carta
            if not m.mesa:
                mask[[ABRIR_FORTE, ABRIR_FRACO]] = True
            else:
                maior = self._maior_da_mesa()
                mata = any(forca(c) > maior for c in m.cartas[AGENTE])
                mask[[MATAR_BARATO, MATAR_FORTE]] = mata
                mask[DEIXAR_FAZER] = True
                mask[ESCONDER] = len(m.rodadas) >= 1   # coberta só a partir da 2ª rodada
            mask[TRUCAR] = self._pode_aumentar(AGENTE)
        return mask

    # ------------------------------------------------- andamento da partida
    def _avancar(self, acao):
        """Roda a partida até a próxima decisão do agente (ou até o fim)."""
        try:
            if acao is None:
                self._decisao = next(self._jogo)
            else:
                self._decisao = self._jogo.send(acao)
        except StopIteration:
            self._fim = True
            self._decisao = None

    def _partida(self):
        primeiro = (self._primeiro_fixo if self._primeiro_fixo is not None
                    else int(self.np_random.integers(4)))
        while max(self.placar) < regras.TENTOS_PARTIDA:
            yield from self._jogar_mao(primeiro)
            self.n_maos += 1
            primeiro = (primeiro + 1) % 4          # a "mão" gira a cada mão

    def _jogar_mao(self, primeiro):
        m = self.mao = _Mao(self._dar_cartas(), primeiro)
        if self.registrar:
            self._evento(f"--- Mão {self.n_maos + 1} | placar {self.placar[0]} x "
                         f"{self.placar[1]} | abre: {NOMES_JOGADORES[primeiro]}")
            for j in range(4):
                cartas = " ".join(nome(c) for c in sorted(m.cartas[j], key=forca, reverse=True))
                self._evento(f"    {NOMES_JOGADORES[j]:<14}{cartas}")

        onze = [p == 11 for p in self.placar]
        if all(onze):                              # 11 a 11: mão de ferro
            self._mao_de_ferro(m)
            return
        if any(onze):                              # mão de 11
            m.mao_de_11 = onze.index(True)
            if m.mao_de_11 == dupla_de(AGENTE):
                m.decidindo_11 = True
                acao = yield DECISAO_ONZE          # o agente vê as cartas do parceiro
                m.decidindo_11 = False
                joga = acao == ACEITAR
            else:
                joga = self.bots[ADV_DIREITA].joga_mao_de_11(
                    m.cartas[ADV_DIREITA] + m.cartas[ADV_ESQUERDA])
            if not joga:
                self._evento(f"Mão de 11: {NOMES_DUPLAS[m.mao_de_11]} correm")
                self._pontuar(1 - m.mao_de_11, regras.CUSTO_CORRER_MAO_DE_11)
                return
            self._evento(f"Mão de 11: {NOMES_DUPLAS[m.mao_de_11]} jogam (vale 3)")
            m.valor = regras.VALOR_MAO_DE_11

        abre = primeiro
        for _ in range(3):
            m.mesa = []
            for k in range(4):
                acabou = yield from self._vez((abre + k) % 4)
                if acabou:                         # alguém correu
                    return
            dupla, jogador = regras.vencedor_da_rodada(
                [(j, 0 if coberta else forca(c)) for j, c, coberta in m.mesa])
            m.rodadas.append(dupla)
            m.mesa = []
            self._evento(f"  {len(m.rodadas)}ª rodada: "
                         + ("empate" if dupla == EMPATE else NOMES_DUPLAS[dupla]))
            vencedor = regras.vencedor_da_mao(m.rodadas)
            if vencedor is not None:
                if vencedor != EMPATE:             # três empates: ninguém pontua
                    self._pontuar(vencedor, m.valor)
                return
            if jogador is not None:                # no empate, abre quem abriu
                abre = jogador

    def _dar_cartas(self):
        """Embaralha e dá 3 cartas para cada um dos 4 jogadores."""
        if self._cartas_fixas:
            return [list(mao) for mao in self._cartas_fixas.pop(0)]
        cartas = baralho()
        ordem = self.np_random.permutation(len(cartas))
        return [[cartas[i] for i in ordem[3 * j:3 * j + 3]] for j in range(4)]

    def _vez(self, j):
        """Vez do jogador j: pode pedir aumento e depois joga uma carta.
        Devolve True se a mão acabou (alguém correu)."""
        m = self.mao
        if j == AGENTE:
            while True:
                acao = yield DECISAO_CARTA
                if acao != TRUCAR:
                    break
                # Trucar antes da carta: o ambiente resolve a resposta dos
                # adversários; se aceitarem, vem um novo estado para a carta.
                if (yield from self._pedir_aumento(AGENTE)):
                    return True
            self._jogar_carta_do_agente(acao)
            return False

        bot = self.bots[j]
        if self._pode_aumentar(j) and bot.quer_trucar(
                m.cartas[j], m.rodadas, dupla_de(j)):
            if (yield from self._pedir_aumento(j)):
                return True
        carta = bot.escolher_carta(m.cartas[j], self._maior_da_mesa())
        self._jogar(j, carta, coberta=False)
        return False

    def _pode_aumentar(self, j):
        m = self.mao
        return (m.mao_de_11 is None                # na mão de 11 ninguém truca
                and m.valor < 12
                and m.direito in (None, dupla_de(j)))

    def _pedir_aumento(self, j):
        """O jogador j pede truco ou aumento. Devolve True se alguém correu.

        Quem corre entrega o valor que a mão tinha antes do pedido.
        """
        m = self.mao
        while True:
            pedido = regras.PROXIMO_VALOR[m.valor]
            self._evento(f"  {NOMES_JOGADORES[j]} pede {'truco' if pedido == 3 else pedido}")
            if dupla_de(j) != dupla_de(AGENTE):
                # Adversários pediram: o agente sempre responde pela dupla.
                m.pendente = (pedido, j)
                acao = yield DECISAO_APOSTA
                m.pendente = None
                if acao == CORRER:
                    self._evento("  Agente corre")
                    self._pontuar(dupla_de(j), m.valor)
                    return True
                self._evento(f"  Agente aceita (vale {pedido})")
                m.valor, m.direito, m.ultimo_pedido = pedido, dupla_de(AGENTE), j
                if acao == ACEITAR:
                    return False
                j = AGENTE                         # Trucar: aceita e pede mais
            else:
                # Nossa dupla pediu: os adversários respondem como dupla.
                vivas = m.cartas[ADV_DIREITA] + m.cartas[ADV_ESQUERDA] + [
                    c for dono, c, _ in m.mesa if dupla_de(dono) == 1]
                if not self.bots[(j + 1) % 4].aceita(vivas):
                    self._evento("  Adversários correm")
                    self._pontuar(dupla_de(j), m.valor)
                    return True
                self._evento(f"  Adversários aceitam (vale {pedido})")
                m.valor, m.direito, m.ultimo_pedido = pedido, 1, j
                return False

    def _jogar_carta_do_agente(self, acao):
        """Traduz a estratégia escolhida para uma carta da mão."""
        m = self.mao
        mao = sorted(m.cartas[AGENTE], key=forca)
        coberta = False
        if acao in (MATAR_BARATO, MATAR_FORTE):
            maior = self._maior_da_mesa()
            vencem = [c for c in mao if forca(c) > maior]
            carta = vencem[0] if acao == MATAR_BARATO else vencem[-1]
        elif acao == ABRIR_FORTE:
            carta = mao[-1]
        elif acao in (ABRIR_FRACO, DEIXAR_FAZER):
            carta = mao[0]
        elif acao == ESCONDER:
            carta, coberta = mao[0], True
        else:
            raise ValueError(f"{NOMES_ACOES[acao]} não é ação de carta")
        self._jogar(AGENTE, carta, coberta)

    def _jogar(self, j, carta, coberta):
        m = self.mao
        m.cartas[j].remove(carta)
        m.mesa.append((j, carta, coberta))
        self._evento(f"  {NOMES_JOGADORES[j]} joga {nome(carta)}"
                     + (" coberta" if coberta else ""))
        m.jogadas[j].append(forca(carta))   # só o agente joga coberta, e ele sabe qual foi

    def _maior_da_mesa(self):
        """Força da maior carta aberta na mesa (0 se a mesa está vazia)."""
        return max((forca(c) for _, c, coberta in self.mao.mesa if not coberta),
                   default=0)

    def _mao_de_ferro(self, m):
        """11 a 11: todos jogam às cegas, na ordem em que receberam as cartas.
        O agente não decide nada; quem vence a mão vence a partida."""
        m.valor = regras.VALOR_MAO_DE_FERRO
        self._evento("Mão de ferro: todos jogam às cegas")
        abre = m.primeiro
        for r in range(3):
            mesa = [((abre + k) % 4, forca(m.cartas[(abre + k) % 4][r]))
                    for k in range(4)]
            dupla, jogador = regras.vencedor_da_rodada(mesa)
            m.rodadas.append(dupla)
            vencedor = regras.vencedor_da_mao(m.rodadas)
            if vencedor is not None:
                if vencedor != EMPATE:
                    self._pontuar(vencedor, m.valor)
                return
            if jogador is not None:
                abre = jogador

    def _pontuar(self, dupla, tentos):
        self.placar[dupla] += tentos
        self._evento(f"  => {NOMES_DUPLAS[dupla]} +{tentos} "
                     f"(placar {self.placar[0]} x {self.placar[1]})")
        self._recompensa += tentos if dupla == dupla_de(AGENTE) else -tentos

    # ------------------------------------------------------------ observação
    def _info(self):
        info = {
            "mascara": self.mascara(),
            "decisao": self._decisao,
            "placar": tuple(min(p, regras.TENTOS_PARTIDA) for p in self.placar),
            "n_maos": self.n_maos,
        }
        if self._fim:
            info["venceu"] = self.placar[0] >= regras.TENTOS_PARTIDA
        return info

    def _obs(self):
        """O que o agente sabe na vez dele, como um vetor de números em [0, 1].

        As cartas entram pela força (1 a 10), porque o naipe não desempata.
        As cartas dos outros jogadores ficam de fora: o problema é
        parcialmente observável (POMDP).
        """
        if self._fim:
            return np.zeros(TAM_OBS, dtype=np.float32)
        m = self.mao

        def por_forca(forcas):
            v = np.zeros(FORCA_MAX, dtype=np.float32)
            for f in forcas:
                v[f - 1] += 1 / 3
            return v

        def um_em(n, i):
            v = np.zeros(n, dtype=np.float32)
            if i is not None:
                v[i] = 1.0
            return v

        minhas = sorted((forca(c) for c in m.cartas[AGENTE]), reverse=True)
        mao_ordenada = np.zeros(3, dtype=np.float32)
        mao_ordenada[:len(minhas)] = np.array(minhas) / FORCA_MAX

        rodadas = np.zeros((3, 4), dtype=np.float32)
        for r in range(3):
            if r >= len(m.rodadas):
                rodadas[r, 3] = 1.0                       # não jogada
            else:
                rodadas[r, {0: 0, 1: 1, EMPATE: 2}[m.rodadas[r]]] = 1.0

        maior = self._maior_da_mesa()
        dono = None
        na_mesa = np.zeros(3, dtype=np.float32)
        for j, c, coberta in m.mesa:
            if coberta:
                continue
            if j != AGENTE:        # a minha carta pode estar na mesa quando
                na_mesa[j - 1] = forca(c) / FORCA_MAX   # respondo a uma aposta
            if dono is None and forca(c) == maior:
                dono = j
        if dono is None:
            dono_da_maior = um_em(3, 0)
        else:
            dono_da_maior = um_em(3, 1 if dupla_de(dono) == dupla_de(AGENTE) else 2)

        if m.pendente:
            valor_pedido = um_em(4, (3, 6, 9, 12).index(m.pendente[0]))
            quem_pediu = um_em(2, 0 if m.pendente[1] == ADV_DIREITA else 1)
        else:
            valor_pedido, quem_pediu = np.zeros(4, np.float32), np.zeros(2, np.float32)

        direito = um_em(3, 0 if m.direito is None else 1 + m.direito)
        ultimo = um_em(5, 0 if m.ultimo_pedido is None else 1 + m.ultimo_pedido)
        placar = np.array([min(p, 12) / 12 for p in self.placar], dtype=np.float32)
        onze = np.array([m.mao_de_11 == 0, m.mao_de_11 == 1, m.decidindo_11],
                        dtype=np.float32)
        parceiro = (por_forca(forca(c) for c in m.cartas[PARCEIRO])
                    if m.mao_de_11 == dupla_de(AGENTE)
                    else np.zeros(FORCA_MAX, np.float32))

        partes = [
            por_forca(minhas), mao_ordenada,
            *(por_forca(m.jogadas[j]) for j in range(4)),
            rodadas.ravel(),
            np.array([maior / FORCA_MAX], dtype=np.float32), dono_da_maior, na_mesa,
            um_em(4, len(m.mesa)),
            um_em(3, min(len(m.rodadas), 2)),
            um_em(5, regras.VALORES.index(m.valor)),
            np.array([m.pendente is not None], dtype=np.float32),
            valor_pedido, quem_pediu, direito, ultimo, placar, onze, parceiro,
            um_em(4, m.primeiro),
            self.mascara().astype(np.float32),
        ]
        return np.concatenate(partes)
