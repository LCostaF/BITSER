# Matrizes NTU de Mus musculus

Fluxo único: ler FASTAs, remover duplicatas, parear por comprimento e comparar as
matrizes de transições mRNA/ncRNA. Quatro módulos: dataset.py (dados e pares),
ntu.py (codificação e matrizes), figures.py (PNG) e run.py (execução).

## Executar

Na raiz do repositório:

    experimental/Daniel/.venv/Scripts/python.exe -X utf8 -B experimental/Daniel/matrizes_ntu/run.py

Em outra máquina, instale as dependências com pip install -r
experimental/Daniel/matrizes_ntu/requirements.txt. O código usa apenas NumPy e Matplotlib.
A .venv existente é local e não faz parte do pacote de distribuição.

Os arquivos esperados são:

    experimental/data_plek/Mus_musculus/sequences/mRNA.fasta
    experimental/data_plek/Mus_musculus/sequences/lncRNA.fasta

--data-dir altera a raiz do PLEK; a espécie permanece Mus musculus.
O pareamento é automático e usa a semente fixa 42.
--output altera a pasta de saída, por padrão results/Mus_musculus/.

## Cálculo

Os rótulos dos cabeçalhos devem corresponder aos arquivos mRNA/lncRNA; lncRNA é
apresentada como ncRNA. A normalização converte U em T, mantém A/C/G/T em maiúsculas
e descarta os demais caracteres.
Duplicatas internas mantêm a primeira ocorrência; sequências compartilhadas
entre classes são excluídas de ambas. O pareamento sorteia sem reposição a mesma
quantidade por comprimento, considerando sequências com pelo menos dez bases.
Cada par é uma tupla (sequência mRNA, sequência ncRNA), mantida apenas em memória.

A codificação usa G < A < T < C, comparação vizinho >= referência e pesos 2**i.
Cada matriz é dividida por L − 9 antes da média: todas as sequências têm o mesmo
peso. As linhas indicam origem e as colunas destino. O suporte estrutural é
obtido enumerando as 4**10 sequências possíveis de dez bases em pequenos lotes.

## Saídas

- suporte_estrutural.png: transições possíveis e impossíveis.
- matrizes_medias.png: médias das duas classes na mesma escala de frequência,
  com fundo branco para transições impossíveis ou não observadas.
- diferenca_classes.png: mRNA − ncRNA em pontos percentuais, com escala centrada em zero
  e destaque para as dez maiores diferenças em módulo.
- parameters.json: localização dos dados e semente fixa do pareamento.

O fluxo recalcula as matrizes e substitui as saídas na pasta escolhida.
Para reproduzir, use os mesmos FASTAs, parâmetros e versões das dependências.
No conjunto atual, a semente 42 produz 2.076 pares.

## Testes essenciais

    experimental/Daniel/.venv/Scripts/python.exe -X utf8 -B -m unittest discover -s experimental/Daniel/matrizes_ntu/tests -v

Cobrem codificação, suporte, normalização das matrizes, peso igual por sequência,
leitura FASTA, deduplicação e correspondência entre pares e registros.

## Referência recuperada

[p0_rook_queen.png](results/p0_rook_queen.png) é o PNG original de
Organização espacial da distribuição teórica P0, recuperado da antiga pasta
results/Mus_musculus/structure/figures/p0_espacial.png, sem alterações.

As barras representam média ± desvio padrão de 200 permutações, não Q1–Q3.
No painel de Geary, as barras de Rook e Queen ocupam aproximadamente 87% e 91%
da altura exibida do eixo Y. Essa altura não representa um limite teórico da métrica.
O PNG é apenas uma referência histórica; run.py gera as três figuras descritas acima.
