# Geração automática de laudo interpretativo de glaucoma a partir do relatório Zeiss Cirrus

## PUC-Rio — MBA em Generative AI & LLM · Turma 2026.1
## PAI — Processamento e Análise de Imagens · Trabalho Final

**Autor:** Rafael Tardelli — tardelli.rafael@gmail.com
**Professora:** Manoela Kohler

Pipeline determinístico que recebe **um PDF de relatório Zeiss Cirrus** (`ONH and RNFL OU
Analysis: Optic Disc Cube 200x200`) e produz um **laudo interpretativo de glaucoma** — o texto
que hoje é escrito à mão pelo médico e que *não existe* na página impressa. Nenhuma rede neural
participa.

> **Aviso clínico:** trabalho acadêmico. As saídas **não** têm validação clínica e não substituem
> a análise do equipamento nem a avaliação de um oftalmologista.

---

## A ideia central

O aparelho de OCT entrega uma página cheia de números e cores. O que ela **não** contém é o
parágrafo interpretativo. Automatizar esse parágrafo produz uma saída que não está no arquivo de
entrada — o oposto de "refazer o exame".

Isso é viável porque **no relatório Zeiss a cor já é o diagnóstico**: cada medida é comparada
contra um banco normativo e o resultado é codificado como cor.

| cor | significado |
|---|---|
| cinza | sem comparação normativa |
| branco | acima do percentil 95 |
| verde | percentis 5–95 — normal |
| amarelo | percentis 1–5 — limítrofe |
| vermelho | abaixo do percentil 1 — fora dos limites |

E a legenda dessa codificação está impressa na própria página, então o classificador **se calibra
a partir da imagem que analisa** — nenhum valor RGB fica escrito no código.

## Onde está cada item da avaliação

| item | seção do notebook |
|---|---|
| Descrição clara do problema abordado | §2 |
| Descrição da base de dados utilizada | §3 |
| Metodologia adotada | §4, implementada nas §6–§10 |
| Relato dos experimentos realizados | §11 |
| Análise dos resultados e conclusões | §15, com a validação em §12 |

## O pipeline

| § | Etapa | Técnica |
|---|---|---|
| 6 | Anonimização e mapeamento | remoção de texto no PDF; imagens embutidas identificadas por tamanho e posição |
| 7 | Calibração cromática | projeções para achar as amostras da legenda; mediana por amostra |
| 8 | Leitura da tabela | projeções para a grade; segmentação de caracteres por projeção vertical; casamento exato de bitmaps |
| 9 | Leitura dos setores | centro e raio por saturação; coordenadas polares; amostragem de cor por ângulo; mapeamento anatômico OD/OE |
| 10 | Escavação horizontal | chave de cor no contorno traçado pelo aparelho; plano de referência; razão de anisotropia |
| 11 | Relato dos experimentos | comparação medida entre as alternativas testadas |
| 12 | Validação | 12 identidades aritméticas internas à página |
| 13 | Consolidação | JSON de achados com procedência |
| 14 | Geração do laudo | regras determinísticas sobre o JSON |

## Resultados

| | |
|---|---|
| Formas distintas na tabela | **49**, de 150 caracteres — byte a byte idênticas |
| Glifos não reconhecidos | **0** na tabela, **0** nos diagramas |
| Validação aritmética | **12/12**, desvio máximo 0,67 µm |
| Escavação horizontal estimada | OD **0,69** · OE **0,68** (o equipamento não a fornece) |
| Laudo gerado | 24 frases, todas rastreáveis até o campo e a cor de origem |
| Margem da classificação de cor | pior caso **8,5×** entre a 1ª e a 2ª referência mais próximas |

### Três achados que sustentam o trabalho

**A fonte do Zeiss é determinística nessa escala.** Os 150 caracteres da tabela colapsam em 49
bitmaps byte a byte idênticos — o casamento de padrões vira consulta a dicionário, sem limiar de
correlação e sem falso positivo possível. É medido, não suposto.

**A página se valida sozinha.** Não há gabarito: a camada de texto do PDF cobre só o cabeçalho.
A validação usa a redundância da própria página — a espessura média aparece na tabela, na média
dos 4 quadrantes e na média dos 12 setores horários, lidas por caminhos independentes do código.

**A anisotropia é estável onde o valor absoluto não é.** Medir C/D absoluto num único B-scan não
reproduz o valor do aparelho, que vem do volume 3D. Mas a razão entre os eixos horizontal e
vertical varia pouco com o plano de referência, e multiplicada pela vertical impressa dá a
escavação horizontal — o único campo do laudo-modelo que o Cirrus não imprime.

## Anonimização

O relatório é de uma paciente real.

| Campo | Regra |
|---|---|
| Nome | **removido do PDF** (texto, não apenas coberto), nunca emitido |
| Identificador | **removido do PDF**, nunca emitido |
| Data de nascimento | preservada no arquivo, **nunca emitida** — só a idade derivada |
| Médico solicitante | fixo `Dr. R2D2`; não vem do exame |
| Identificação no laudo | `Paciente <carimbo de tempo da execução>` |

A remoção usa `add_redact_annot` + `apply_redactions` do PyMuPDF, que apagam os glifos em vez de
desenhar um retângulo por cima, e zera os metadados do documento. As 22 imagens embutidas ficam
byte a byte intactas.

O PDF original fica em `imagens_originais/`, que está no `.gitignore`. O versionado é
`imagens/optic_disc_cube_anon.pdf`.

## Por que não uma rede neural

Além de a interpretação ser um problema de regras, uma versão anterior deste projeto avaliou o
classificador RETFound pré-treinado para OCT. Sondas de controle mostraram que ele devolvia a
mesma classe para qualquer entrada — ruído branco 0,431, cinza uniforme 0,332, imagem toda branca
0,696 — e a análise dos pesos confirmou uma cabeça de classificação **nunca treinada**. O
resultado está resumido na §15 do notebook.

Aqui, cada afirmação do laudo pode ser apontada de volta a um pixel da página. Nenhum
classificador opaco ofereceria isso, mesmo funcionando.

## Estrutura

```
pai-analise_retina-rafael_tardelli.ipynb   # o trabalho, 15 seções
imagens/optic_disc_cube_anon.pdf           # única entrada, anonimizada
imagens_originais/                         # originais — gitignored
requirements.txt
```

## Instalação e execução

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name pai-retina --display-name "Python (PAI Retina)"
jupyter lab pai-analise_retina-rafael_tardelli.ipynb
```

Sem dependência de aprendizado de máquina: `pymupdf`, `numpy`, `scipy`, `pillow`, `pandas`,
`matplotlib`. Execução completa em segundos.

Se `imagens_originais/` não existir — o caso de quem clona o repositório — a §6 apenas confere
que o PDF versionado já está anonimizado e segue normalmente.

## Limitações

- **Um único exame.** O layout é fixo para este protocolo e versão de software; a robustez a
  outras versões não foi testada.
- **A escavação horizontal é estimativa**, obtida de um único corte e não do volume. O laudo
  declara isso.
- **A rotulação do banco de formas** depende de uma transcrição manual feita uma vez. É auditável
  — as 49 formas são exibidas na §8.5 — mas não é automática.
- **Não há GCC**: o protocolo Optic Disc Cube não cobre a mácula. O laudo declara a ausência.
- **A profundidade axial de 2 mm** é assumida; seu efeito está quantificado na §10.3.
- **O laudo não é um diagnóstico.**

## Extensão natural

O relatório *Ganglion Cell OU Analysis* usa a mesma gramática visual — mesma grade, mesmos
diagramas setoriais, mesma legenda, mesma fonte. Quase todo o código se aplica, e ele traz o
**GCC**, o campo que falta ao laudo.

## Material

Relatório `ONH and RNFL OU Analysis: Optic Disc Cube 200x200`, Cirrus HD-OCT, Carl Zeiss Meditec,
software 7.0.3.19. A codificação normativa por cor e os percentis 95/5/1 estão impressos na
própria página, no bloco "Diversified: Distribution of Normals".
