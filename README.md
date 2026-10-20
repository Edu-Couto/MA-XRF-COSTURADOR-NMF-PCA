# MA-XRF — Costura, NMF e PCA

Códigos Python utilizados na tese de doutorado de Eduardo Couto, desenvolvida no Programa de Pós-Graduação em Física da Universidade do Estado do Rio de Janeiro (UERJ).

O trabalho integra dados de macrofluorescência de raios X (MA-XRF), fatoração de matrizes não negativas com divergência de Kullback–Leibler (NMF-KL) e análise por componentes principais (PCA) para a comparação químico-material de pinturas.

## Programas

| Arquivo | Etapa da pesquisa |
|---|---|
| `costurador.py` | Reconstituição dos mapeamentos adquiridos em diferentes varreduras espaciais. |
| `NMF-Final.py` | Análise dos dados por NMF-KL. |
| `PCA-COM GRAFICO SINTESE 3D.py` | Análise por PCA e elaboração das representações gráficas. |

## Organização da metodologia

A pesquisa compreende a reconstituição dos mapeamentos MA-XRF, a análise individual das pinturas por NMF-KL, a interpretação e classificação das componentes em famílias químico-espectrais e a construção de uma matriz comum para a comparação das obras por PCA.

A classificação das componentes em famílias constitui uma etapa de interpretação dos resultados. A correspondência entre essas famílias permite organizar as informações das diferentes pinturas em uma base comum.

## Relação com a tese

Este repositório reúne os códigos associados à metodologia descrita na tese. Os critérios de análise, as decisões de interpretação e a discussão dos resultados são apresentados no texto da pesquisa.

## Documentação de execução

A documentação de cada programa deve identificar:

- a versão do Python e das bibliotecas utilizadas;
- os arquivos de entrada e sua organização;
- os parâmetros definidos pelo usuário;
- o procedimento de execução;
- os arquivos produzidos.

## Como citar

Ao utilizar estes códigos, cite a versão consultada do repositório. As informações de autoria e citação serão registradas no arquivo `CITATION.cff`.
