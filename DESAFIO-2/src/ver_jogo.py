"""Material extra: mostra como o DQN escolhe uma jogada usando Q e máscara.

    python src/ver_jogo.py
    python src/ver_jogo.py --modelo resultados/treino_final/modelo.pt
"""
import argparse
from pathlib import Path
from dqn import AgenteDQN
from truco.ambiente import TrucoEnv, NOMES_ACOES

RAIZ = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelo', type=Path, default=RAIZ/'resultados/treino_final/modelo.pt')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--decisoes', type=int, default=4)
    args = parser.parse_args()

    agente = AgenteDQN()
    agente.carregar(args.modelo)
    jogo = TrucoEnv(registrar=True)
    estado, info = jogo.reset(seed=args.seed)
    print('Modelo:', args.modelo)
    print('A rede prevê 9 Q-valores; só ações VÁLIDAS podem ser escolhidas.')
    for i in range(args.decisoes):
        q = agente.valores(estado)
        acao = agente.escolher(estado, info['mascara'], explorar=False)
        print(f'\nDecisão {i+1} — {info["decisao"]}, placar {info["placar"]}')
        for a, nome in enumerate(NOMES_ACOES):
            if info['mascara'][a]:
                print(f'  {nome:18} Q={q[a]:+.3f}')
        print('  -> Escolheu:', NOMES_ACOES[acao])
        estado, recompensa, terminou, _, info = jogo.step(acao)
        print('  Recompensa:', recompensa, 'tentos')
        if terminou:
            print('  Fim da partida')
            break


if __name__ == '__main__':
    main()
