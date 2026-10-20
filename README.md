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

Este repositório reúne os códigos associados à metodologia descrita na tese. Os procedimentos, os parâmetros, os critérios de interpretação e a discussão dos resultados são apresentados no texto da pesquisa.

## Ambiente de execução

Os códigos foram utilizados nas análises da tese com Python 3.13. As dependências estão listadas no arquivo `requirements.txt`, sem versões fixadas.

A execução dos aplicativos pode exigir ajustes de compatibilidade entre as bibliotecas e o ambiente do usuário.

## Instalação

Baixe o repositório e extraia os arquivos em uma pasta. Com o terminal aberto nessa pasta, instale as dependências:

```bash
python -m pip install -r requirements.txt
```

Os programas possuem interface gráfica e devem ser executados em um computador com ambiente desktop. A instalação do Python deve incluir suporte a Tkinter, utilizado pelo NMF e pela PCA.

## Execução

Execute separadamente o comando correspondente ao programa desejado.

**Costurador:**

```bash
python costurador.py
```

**NMF:**

```bash
python NMF-Final.py
```

**PCA:**

```bash
python "PCA-COM GRAFICO SINTESE 3D.py"
```

O costurador e o programa NMF trabalham com dados em HDF5. O programa PCA recebe uma matriz CSV organizada com uma obra por linha e variáveis espectrais por coluna.

A preparação dos dados e os critérios de análise são descritos na metodologia da tese.

## Como citar

Ao utilizar estes códigos em trabalhos acadêmicos, solicita-se a citação do repositório, identificando a versão ou o commit consultado. As informações de autoria e citação estão disponíveis no arquivo `CITATION.cff`.

## Licença

Os códigos deste repositório são distribuídos sob a GNU General Public License, versão 3 (GPL-3.0-only). Consulte o arquivo `LICENSE` para conhecer os termos.
