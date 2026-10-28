# Agente de Truco Mineiro com Aprendizado por Reforço

Alunos: Matheus Cruz e Tarik

## Desafio 1

Apresentação: [Slides/Agente-Truco.pdf](Slides/Agente-Truco.pdf)

Um agente que joga truco mineiro 2 contra 2, com baralho vazio. O agente controla um jogador; o parceiro e os adversários são bots com regras fixas.

- **Estado:** cartas na mão, cartas jogadas, quem venceu cada rodada, maior carta na mesa, valor da mão e placar.
- **Ações:** estratégias em vez de cartas: matar barato, matar forte, deixar fazer, esconder, abrir forte, abrir fraco, trucar, aceitar e correr.
- **Recompensa:** tentos ganhos menos tentos perdidos pela dupla, ao fim de cada mão.
