"""1 de 3: a rede RECEBE 127 números e DEVOLVE 9 valores Q."""
from torch import nn


class RedeNeural(nn.Module):
    def __init__(self, entradas=127, saidas=9, neuronios=128):
        super().__init__()
        # Duas camadas com ReLU e uma camada final linear.
        self.head = nn.Sequential(
            nn.Linear(entradas, neuronios), nn.ReLU(),
            nn.Linear(neuronios, neuronios), nn.ReLU(),
            nn.Linear(neuronios, saidas),
        )

    def forward(self, estado):
        return self.head(estado)  # nao são probabilidades!
