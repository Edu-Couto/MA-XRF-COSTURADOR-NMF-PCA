# MA-XRF: Costurador, NMF-KL e PCA

Repositório destinado aos códigos e à documentação associados à metodologia de processamento de dados MA-XRF e comparação químico-material entre pinturas desenvolvida na tese de doutorado.

## Conteúdo disponível

| Programa | Finalidade | Documentação |
|---|---|---|
| **Júlia1 — Costurador** | Reconstituição de cubos MA-XRF a partir de varreduras parciais com sobreposição. | [Instruções e descrição do processamento](Julia1/README.md) |

Os códigos da NMF-KL e da PCA serão acrescentados após a organização de suas versões utilizadas no trabalho.

## Costurador Júlia1

O diretório [Julia1](Julia1/) contém o programa `costurador_Julia1.py`, a documentação de uso, um trecho da metodologia em LaTeX e o arquivo de verificação de integridade do código.

O processamento reúne o alinhamento dos blocos, a equalização relativa de intensidade e a composição do mosaico por uma linha de costura. Os detalhes, parâmetros e esclarecimentos sobre os textos originais do programa estão no README do diretório Julia1.

As instruções de instalação e execução estão em [Julia1/README.md](Julia1/README.md).
