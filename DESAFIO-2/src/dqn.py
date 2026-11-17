"""2 de 3: o agente escolhe ações e APRENDE comparando previsão e alvo.

O Truco não está aqui. O ambiente entrega um estado, máscara e recompensa.
Ao lado do treino, leia apenas escolher(), guardar() e aprender().
"""
import numpy as np
import torch
from torch import nn
from rede_neural import RedeNeural


class AgenteDQN:
    def __init__(self, entradas=127, acoes=9, seed=1):
        self.sorteador = np.random.default_rng(seed)
        self.epsilon = 1.0  # chance de EXPLORAR; no treino, vai caindo
        self.gamma = 0.99   # quanto consideramos o valor futuro
        self.lote = 64      # experiências por atualização
        self.aquecimento = 1000  # só aprender após ter experiências suficientes
        self.a_cada = 1000      # sincronizar rede-alvo após 1000 updates
        self.updates = 0
        self.online = RedeNeural(entradas, acoes)  # esta rede aprende
        self.alvo = RedeNeural(entradas, acoes)    # esta calcula o alvo
        self.alvo.load_state_dict(self.online.state_dict())
        self.otimizador = torch.optim.Adam(self.online.parameters(), lr=0.00025)
        self.perda = nn.SmoothL1Loss()  # Huber
        self.memoria = []  # lista circular: sorteio rápido mesmo com 100 mil experiências
        self.proxima_posicao = 0
        self.entradas, self.acoes = entradas, acoes

    def valores(self, estado):
        """Transforma um estado em 9 Q-valores."""
        with torch.no_grad():
            x = torch.tensor(np.asarray(estado), dtype=torch.float32).unsqueeze(0)
            return self.online(x).squeeze(0).numpy()

    def escolher(self, estado, mascara, explorar=True):
        """Escolhe ação aleatória (explorar) ou maior Q VÁLIDO (aproveitar)."""
        validas = np.flatnonzero(mascara)
        if explorar and self.sorteador.random() < self.epsilon:
            return int(self.sorteador.choice(validas))
        q = self.valores(estado)
        q[~np.asarray(mascara, dtype=bool)] = -np.inf
        return int(q.argmax())

    def guardar(self, estado, acao, recompensa, proximo, terminou, mascara_proxima):
        """Guarda uma experiência: estado, ação, recompensa, próximo estado etc."""
        experiencia = (np.array(estado, copy=True), int(acao), float(recompensa),
                       np.array(proximo, copy=True), bool(terminou),
                       np.array(mascara_proxima, copy=True))
        if len(self.memoria) < 100_000:
            self.memoria.append(experiencia)
        else:
            self.memoria[self.proxima_posicao] = experiencia
        self.proxima_posicao = (self.proxima_posicao + 1) % 100_000

    def aprender(self):
        """Uma atualização: amostras -> Q previsto -> alvo -> erro -> Adam."""
        if len(self.memoria) < max(self.aquecimento, self.lote):
            return None

        # 1) Sortear experiências de partidas anteriores.
        indices = self.sorteador.integers(0, len(self.memoria), size=self.lote)
        experiencias = [self.memoria[int(i)] for i in indices]
        estados, acoes, recompensas, proximos, finais, mascaras = zip(*experiencias)
        s = torch.tensor(np.stack(estados), dtype=torch.float32)
        a = torch.tensor(acoes, dtype=torch.int64).unsqueeze(1)
        r = torch.tensor(recompensas, dtype=torch.float32)
        s2 = torch.tensor(np.stack(proximos), dtype=torch.float32)
        fim = torch.tensor(finais, dtype=torch.bool)
        mask2 = torch.tensor(np.stack(mascaras), dtype=torch.bool)

        # 2) O que a rede ONLINE prevê para a ação que realmente foi tomada?
        previsto = self.online(s).gather(1, a).squeeze(1)

        # 3) O que ela deveria prever? Bellman: r + gamma * Q_futuro.
        with torch.no_grad():
            q_futuro = self.alvo(s2).masked_fill(~mask2, -torch.inf)
            melhor_futuro = q_futuro.max(dim=1).values
            melhor_futuro = torch.where(fim, 0.0, melhor_futuro)
            esperado = r + self.gamma * melhor_futuro

        # 4) Quanto errou? Backpropagation + Adam corrigem os pesos ONLINE.
        erro = self.perda(previsto, esperado)
        self.otimizador.zero_grad()
        erro.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.otimizador.step()

        # 5) A rede-ALVO recebe uma cópia dos pesos de tempos em tempos.
        self.updates += 1
        if self.updates % self.a_cada == 0:
            self.alvo.load_state_dict(self.online.state_dict())
        return float(erro.item())

    def salvar(self, caminho):
        torch.save({'online': self.online.state_dict(), 'obs_dim': self.entradas,
                    'n_actions': self.acoes, 'algorithm': 'dqn'}, caminho)

    def carregar(self, caminho):
        dados = torch.load(caminho, map_location='cpu', weights_only=True)
        self.online.load_state_dict(dados['online'])
        self.alvo.load_state_dict(dados['online'])
