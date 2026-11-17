import numpy as np
import torch
from dqn import AgenteDQN
from rede_neural import RedeNeural
from truco.ambiente import TrucoEnv, TAM_OBS, N_ACOES


def test_formato_rede():
    net = RedeNeural(TAM_OBS, N_ACOES)
    assert net(torch.zeros(1, TAM_OBS)).shape == (1, N_ACOES)


def test_nunca_escolhe_acao_proibida():
    agente = AgenteDQN(entradas=4, acoes=5)
    estado = np.ones(4, np.float32)
    mascara = np.array([False, True, False, True, False])
    for epsilon in (0.0, 0.3, 1.0):
        agente.epsilon = epsilon
        assert {agente.escolher(estado, mascara) for _ in range(80)} <= {1, 3}


def test_aprender_modifica_pesos():
    torch.manual_seed(0)
    agente = AgenteDQN(entradas=2, acoes=3)
    agente.aquecimento = 0
    agente.lote = 8
    antes = [x.detach().clone() for x in agente.online.parameters()]
    for _ in range(25):
        agente.guardar([1, 0], 0, 1, [0, 0], True, [False, False, False])
    perda = agente.aprender()
    assert perda is not None and np.isfinite(perda)
    assert any(not torch.equal(p, old) for p, old in zip(agente.online.parameters(), antes))


def test_salva_e_carrega_modelo_novo(tmp_path):
    caminho = tmp_path / "modelo.pt"
    agente = AgenteDQN()
    agente.salvar(caminho)
    agente_novo = AgenteDQN()
    agente_novo.carregar(caminho)
    estado, info = TrucoEnv().reset(seed=7)
    assert np.allclose(agente.valores(estado), agente_novo.valores(estado))
    assert info['mascara'][agente_novo.escolher(estado, info['mascara'], False)]
