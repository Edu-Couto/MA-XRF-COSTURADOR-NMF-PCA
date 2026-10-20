# Costurador de cubos MA-XRF — Júlia 1 

Este diretório reúne o código utilizado para reconstituir cubos de dados de macrofluorescência de raios X (MA-XRF) a partir de varreduras parciais com sobreposição.

O arquivo `costurador_Júlia1.py` preserva integralmente o conteúdo do original. Apenas o nome do arquivo foi padronizado para uso neste diretório. A interface, os comentários, os nomes internos, os parâmetros e as rotinas de processamento foram mantidos.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `costurador_Júlia1.py` | Código original, com interface gráfica PyQt5. |
| `docs/metodologia_costurador.tex` | Descrição metodológica conferida com as rotinas do código. Usa as referências e os rótulos da tese; é um trecho para inclusão no documento principal. |
| `SHA256SUMS.txt` | Resumo SHA-256 do código, para verificar a integridade do arquivo. |

## Dependências e execução

As dependências externas importadas pelo programa são **PyQt5, h5py, NumPy e SciPy**.

Para instalá-las em um ambiente Python adequado:

```bash
python -m pip install PyQt5 h5py numpy scipy
```

Dentro deste diretório, execute:

```bash
python costurador_Júlia1.py
```

O cabeçalho original menciona o nome `xrf_stitcher.py`. Neste diretório, o comando acima corresponde ao nome efetivamente fornecido.

O script não informa as versões exatas de Python e das bibliotecas usadas no processamento da tese. A lista de dependências acima identifica as bibliotecas necessárias, sem fixar versões. A reprodução do ambiente original requer o registro das versões presentes no computador utilizado para o processamento.

## Dados de entrada

Cada entrada deve ser um arquivo HDF5 com um dataset `/cube`, organizado como `(Y, X, Z)`: linhas, colunas e canais espectrais. Os dois cubos devem representar canais correspondentes em energia. O programa verifica a igualdade do número de canais antes da gravação do mosaico.

A interface desta versão utiliza a soma das contagens no eixo espectral para construir as imagens de registro. O processamento é feito com dois arquivos por vez. Para reunir mais de duas varreduras, o arquivo reconstituído pode ser utilizado como entrada em uma costura seguinte.

## Uso da interface

1. Selecione a direção de costura, vertical ou horizontal.
2. Carregue os dois arquivos HDF5, mantendo a ordem e a orientação dos blocos utilizadas no processamento.
3. Clique em **Analisar Sobreposição e Qualidade**. Os parâmetros avançados permitem ajustar a busca da sobreposição e do deslocamento transversal.
4. Escolha o mapa que será a referência de escala. A interface apresenta uma sugestão baseada no indicador denominado SNR no código.
5. Defina o caminho do HDF5 de saída. Use um nome distinto para preservar resultados anteriores: o programa tenta substituir uma saída já existente com o mesmo nome.
6. Clique em **Normalizar e Costurar**. O relatório HTML é salvo com o mesmo nome-base e na mesma pasta do HDF5.

## Procedimento implementado

O registro compara faixas nas bordas dos blocos, examinando extensões candidatas de sobreposição. O critério de escolha utiliza a correlação normalizada e um termo adicional que favorece sobreposições mais extensas. O deslocamento transversal é aplicado como uma translação inteira, sem interpolação espacial dos espectros.

A estimativa inicial do fator de equalização utiliza as faixas de registro. O refinamento utiliza uma amostra regular de pares de pixels correspondentes na interseção dos cubos alinhados. Para cada par, são somadas as contagens ao longo do eixo espectral e calculada a razão entre as intensidades totais do primeiro e do segundo bloco. São utilizados pares com intensidades finitas e positivas. Havendo pelo menos 50 razões válidas, são mantidas as razões entre os percentis 1 e 99. O fator refinado é a mediana das razões consideradas. A rotina mantém a estimativa inicial se ocorrer uma exceção no refinamento; situações sem amostras válidas podem resultar em fator unitário.

O mapa escolhido pelo usuário define a referência de escala. Se o primeiro mapa é a referência, o segundo recebe o fator estimado; se o segundo é a referência, o primeiro recebe o inverso desse fator. A multiplicação utiliza um fator comum a todos os canais de cada bloco.

O recorte transversal delimita a extensão comum aos blocos: em X para costuras verticais e em Y para costuras horizontais. Essa operação pode excluir pixels válidos presentes em apenas um dos blocos.

A região sobreposta é dividida por uma linha de costura calculada por programação dinâmica, com custo associado às diferenças entre as intensidades integradas e à continuidade espacial do traçado. O caminho recebe uma suavização por mediana móvel. Cada posição do mosaico recebe dados de apenas uma aquisição, após a equalização; não são calculadas médias entre espectros dos dois blocos.

## Parâmetros presentes nesta versão

Os valores abaixo são os que constam no arquivo. Os parâmetros de busca podem ser modificados pela interface e não constituem um registro dos valores efetivamente usados em cada obra.

| Parâmetro | Valor no código |
|---|---|
| Tipo de dado de saída | `OUTPUT_DTYPE_MODE = "uint32"` |
| Compressão | GZIP, nível 4, com `shuffle` |
| Busca: extensão inicial | 10 pixels |
| Busca: extensão máxima | 300 pixels |
| Busca: passo | 1 pixel |
| Busca: deslocamento transversal máximo | 300 pixels |
| Amostragem do refinamento | `stride = 2`; parâmetro `max_samples = 200000`, com ajuste do espaçamento |
| Corte das razões no refinamento | Percentis 1 e 99, com pelo menos 50 razões válidas |
| Penalidade de continuidade da linha de costura | 0,20 |
| Janela da mediana móvel da linha de costura | 9 posições |

## Saídas

O HDF5 contém `/cube`, `/x_axis`, `/y_axis`, `/energy` e o grupo `/metadata`. Os eixos espaciais são gravados como índices de pixels. O dataset `/energy` contém **índices dos canais**, e não uma calibração energética em keV.

O grupo `/metadata` registra informações como arquivos de origem, direção, deslocamentos, extensão da sobreposição, referência de escala, fatores aplicados, formato numérico, compressão e data de processamento. O caminho da linha de costura é salvo, quando disponível, em um dataset na raiz chamado `/optimal_seam_y_by_x` ou `/optimal_seam_x_by_y`.

Na configuração `uint32`, os valores reescalados são arredondados e limitados ao intervalo representável antes da conversão. Essa gravação pode alterar as proporções exatas entre canais após o reescalamento. A configuração original foi preservada neste arquivo.

## Esclarecimentos sobre os textos originais do programa

O arquivo conserva comentários e mensagens do período de desenvolvimento. A tabela abaixo esclarece descrições antigas presentes no cabeçalho, na interface ou no relatório HTML, com base nas instruções executadas por esta versão.

| Texto ou rótulo original | Esclarecimento |
|---|---|
| Normalização por contagem total, com a expressão `I_norm = I / Σ_z I` | A costura utiliza um fator multiplicativo entre os blocos, estimado na sobreposição. |
| `mediana(A)/mediana(B)` | Essa forma é utilizada na estimativa inicial por faixas. No refinamento da interseção dos cubos, utiliza-se a mediana das razões `A/B` de pixels correspondentes. |
| Mapa a manter na sobreposição | A escolha define a referência de escala. A origem dos dados na sobreposição é definida pela linha de costura. |
| Afirmação de que a saída é sempre `float32` | O parâmetro efetivo nesta versão é `uint32`; o campo correspondente nos metadados registra o formato utilizado. |
| Recorte descrito como opcional | O recorte transversal é automático no caminho utilizado pela interface. Esta interface não apresenta uma opção para desativá-lo. |
| Indicador SNR | O código calcula a razão entre a média e o desvio padrão das contagens de todo o cubo. |
| Comentário sobre compressão máxima ou taxa de aproximadamente dez vezes | O nível configurado é 4; a taxa efetiva de compressão depende dos dados. |

## Referência para o princípio de equalização relativa

Ban, S.; Kim, T. (2022). *Relative Radiometric Calibration of UAV Images for Image Mosaicking*. The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, XLIII-B1-2022, 361–366. DOI: https://doi.org/10.5194/isprs-archives-XLIII-B1-2022-361-2022.

Essa referência discute a estimação de ganhos a partir de regiões sobrepostas. O estimador por mediana das razões descrito aqui corresponde à implementação deste código. No trecho LaTeX, a referência é identificada pela chave `ban2022`, que deve existir no arquivo bibliográfico da tese.
