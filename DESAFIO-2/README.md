# DQN no Truco Mineiro

Um agente **Deep Q-Network (DQN)** toma decisões em partidas de Truco Mineiro 2 contra 2. O ambiente, as regras do jogo e os bots possuem lógica própria; a DQN aprende a escolher entre **9 ações discretas** a partir de um vetor com **127 características** do estado.

## Estrutura

- `src/rede_neural.py` — rede MLP (127 → 128 → ReLU → 128 → ReLU → 9).
- `src/dqn.py` — seleção de ações, memória de experiências e atualização dos pesos.
- `src/treinar.py` — treinamento, avaliação e geração dos gráficos.
- `src/ver_jogo.py` — demonstração das decisões do modelo salvo.
- `src/truco/`, `src/bots/` e `src/auxiliares/` — regras, ambiente e políticas auxiliares.
- `testes/` — testes automatizados.
- `resultados/treino_final/` — pesos, dados e gráficos do treinamento incluído.

## Instalação

Requer Python com as dependências de `requirements.txt`. Na raiz do repositório:

```bash
python -m pip install -r requirements.txt
python -m pytest -q testes
```

## Executar o modelo treinado

```bash
python src/ver_jogo.py
```

## Treinar uma rede nova

Exemplo curto, para aprender como o programa funciona:

```bash
python src/treinar.py --rapido --saida resultados/teste_rapido
```

Reproduzir a configuração do experimento incluído (treina uma rede nova, sem carregar pesos anteriores):

```bash
python src/treinar.py --episodios 5000 --seed 42 --avaliacao-partidas 200 --final-partidas 2000 --saida resultados/minha_tentativa
```

Use outra pasta em `--saida` para **não sobrescrever os resultados incluídos** em `resultados/treino_final/`.

## Resultados incluídos

Treinamento de **5.000 partidas** com seed **42**. A avaliação final, sem exploração, alcançou **79,4% de vitórias em 2.000 partidas novas** contra os bots do ambiente. A recompensa média foi de **−6,73 tentos nos 200 primeiros episódios** e **+4,07 tentos nos 200 últimos**.

Arquivos disponíveis em `resultados/treino_final/`:

| Arquivo | Conteúdo |
| --- | --- |
| `modelo.pt` | Pesos do modelo treinado nesta versão |
| `treino.csv` | Recompensa e epsilon por partida |
| `avaliacao.csv` | Avaliações periódicas sem exploração |
| `final.json` | Resultado da avaliação final |
| `config.json` | Parâmetros usados no treinamento |
| `grafico_recompensa.png` | Evolução da recompensa média |
| `grafico_vitorias.png` | Evolução da taxa de vitórias |

O experimento utiliza uma única seed e adversários fixos. O resultado **não representa** desempenho contra jogadores humanos ou bots diferentes.

## Como o agente aprende

A rede estima valores Q para as 9 ações; uma máscara impede ações inválidas. Durante o treinamento, uma política epsilon-greedy equilibra exploração e aproveitamento do que foi aprendido. O DQN armazena transições em um replay buffer, usa uma rede-alvo para formar o alvo temporal e atualiza os pesos da rede online com perda Huber e otimizador Adam.
