"""3 de 3: manda jogar Truco, guarda experiências e chama aprender().

Demo rápida:     python src/treinar.py --rapido
Treino escolhido: python src/treinar.py --episodios 5000 --seed 42

A execução cria 2 gráficos: evolução da recompensa e taxa de vitórias.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from dqn import AgenteDQN
from truco.ambiente import TrucoEnv

RAIZ = Path(__file__).resolve().parent.parent


def avaliar(agente, env, partidas=100, primeira_seed=1_000_000):
    """Sem exploração e sem atualizar pesos: mede a política atual."""
    vitorias = 0
    for seed in range(primeira_seed, primeira_seed + partidas):
        estado, info = env.reset(seed=seed)
        terminou = False
        while not terminou:
            acao = agente.escolher(estado, info['mascara'], explorar=False)
            estado, _, terminou, _, info = env.step(acao)
        vitorias += bool(info['venceu'])
    return vitorias / partidas


def desenhar_graficos(registros, avaliacoes, pasta):
    """Desenha a recompensa média e a avaliação da própria execução."""
    episodios = [linha[0] for linha in registros]
    recompensas = np.array([linha[1] for linha in registros], dtype=float)
    janela = min(200, len(recompensas))
    media = np.convolve(recompensas, np.ones(janela)/janela, mode='valid')
    plt.figure(figsize=(8, 4))
    plt.plot(episodios[janela-1:], media)
    plt.xlabel('Partidas de treinamento')
    plt.ylabel('Recompensa média (tentos)')
    plt.title(f'Evolução da recompensa ({janela} partidas)')
    plt.tight_layout()
    plt.savefig(pasta / 'grafico_recompensa.png', dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot([a[0] for a in avaliacoes], [a[1]*100 for a in avaliacoes], marker='o')
    plt.ylim(0, 100)
    plt.xlabel('Partidas de treinamento')
    plt.ylabel('Vitórias na avaliação (%)')
    plt.title('DQN: avaliação sem exploração')
    plt.tight_layout()
    plt.savefig(pasta / 'grafico_vitorias.png', dpi=140)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Treinamento didático da DQN no Truco')
    parser.add_argument('--rapido', action='store_true', help='300 partidas para estudar')
    parser.add_argument('--episodios', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--avaliacao-partidas', type=int, default=200,
                        help='quantas partidas em cada ponto da curva (padrão: 200)')
    parser.add_argument('--final-partidas', type=int, default=2000,
                        help='partidas separadas para avaliação final (padrão: 2000)')
    parser.add_argument('--saida', type=Path, default=None)
    args = parser.parse_args()
    if args.rapido:
        args.episodios = 300
        if args.final_partidas == 2000:
            args.final_partidas = 100  # demonstração rápida, não resultado definitivo
    if args.episodios <= 0 or args.final_partidas <= 0 or args.avaliacao_partidas <= 0:
        parser.error("--episodios, --avaliacao-partidas e --final-partidas devem ser positivos")
    if args.saida is None:
        args.saida = RAIZ / 'resultados' / ('teste_rapido' if args.rapido else 'treino_final')
    args.saida.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    ambiente = TrucoEnv()
    ambiente_avaliacao = TrucoEnv()
    agente = AgenteDQN(seed=args.seed)
    total_passos = 0
    treino = []
    avaliacoes = []
    intervalo = 100 if args.rapido else 500
    partidas_avaliacao = 100 if args.rapido else args.avaliacao_partidas
    rampa_epsilon = 150 if args.rapido else 5000

    # Uma PARTIDA inteira = um episódio; um PASSO = uma decisão do agente.
    for ep in range(1, args.episodios + 1):
        agente.epsilon = max(0.05, 1 - 0.95 * (ep - 1) / rampa_epsilon)
        estado, info = ambiente.reset(seed=args.seed * 10_000_000 + ep)
        terminou = False
        recompensa_partida = 0

        while not terminou:
            # 1) Observar, escolher e agir no jogo.
            acao = agente.escolher(estado, info['mascara'])
            proximo, recompensa, terminou, _, nova_info = ambiente.step(acao)

            # 2) Guardar a experiência (recompensa / 12 para treinar).
            agente.guardar(estado, acao, recompensa / 12, proximo,
                          terminou, nova_info['mascara'])
            recompensa_partida += recompensa
            estado, info = proximo, nova_info
            total_passos += 1

            # 3) Corrigir os pesos de 4 em 4 passos (após 1.000 experiências).
            if total_passos % 4 == 0:
                agente.aprender()

        treino.append((ep, recompensa_partida, agente.epsilon))
        if ep == 1 or ep % intervalo == 0 or ep == args.episodios:
            taxa = avaliar(agente, ambiente_avaliacao, partidas_avaliacao)
            avaliacoes.append((ep, taxa))
            print(f'Partida {ep:5} | epsilon={agente.epsilon:.2f} '
                  f'| vitórias={taxa:.1%} | retorno={recompensa_partida:+.0f}', flush=True)

    # 4) Avaliar em partidas NOVAS e guardar os critérios de reprodução.
    resultado_final = avaliar(agente, ambiente_avaliacao, args.final_partidas,
                             primeira_seed=2_000_000)
    (args.saida / 'final.json').write_text(
        json.dumps({'semente_treinamento': args.seed, 'episodios': args.episodios,
                    'partidas': args.final_partidas, 'taxa_vitoria': resultado_final,
                    'proveniencia': 'Treinado nesta versão simplificada'},
                   indent=2, ensure_ascii=False), encoding='utf-8')
    (args.saida / 'config.json').write_text(
        json.dumps({'seed': args.seed, 'episodios': args.episodios,
                    'partidas_avaliacao_final': args.final_partidas, 'partidas_por_avaliacao_periodica': partidas_avaliacao, 'rampa_epsilon': rampa_epsilon,
                    'recompensa_escala': '1/12', 'gamma': agente.gamma,
                    'lote': agente.lote, 'aquecimento': agente.aquecimento,
                    'aprende_a_cada_passos': 4, 'sincronia_rede_alvo_updates': agente.a_cada},
                   indent=2, ensure_ascii=False), encoding='utf-8')

    # 5) Salvar resultados e os 2 gráficos do treinamento NOVO.
    agente.salvar(args.saida / 'modelo.pt')
    for nome, cabecalho, linhas in (
        ('treino.csv', ('episodio', 'recompensa', 'epsilon'), treino),
        ('avaliacao.csv', ('episodio', 'taxa_vitoria'), avaliacoes),
    ):
        with open(args.saida / nome, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(cabecalho)
            writer.writerows(linhas)
    desenhar_graficos(treino, avaliacoes, args.saida)
    print(f'\nAvaliação final: {resultado_final:.1%} em {args.final_partidas} partidas novas')
    print('Pronto! Veja os gráficos em:', args.saida)


if __name__ == '__main__':
    main()
