# Análise de Retina com RETFound
## PUC-RIO: MBA Gen AI & LLM
## PAI - Processamento e Análise de Imagens - Trabalho Final

**Autor:** Rafael Tardelli — tardelli.rafael@gmail.com

Pipeline de inferência em Jupyter Notebook que executa dois modelos RETFound pré-treinados (ViT-Large/16) sobre imagens de retina. O notebook não realiza treinamento nem fine-tuning — apenas inferência. Cada modalidade de imagem (OCT B-scan ou retinografia colorida) é roteada ao modelo correspondente, indicadores de confiança são calculados deterministicamente e o resultado é gravado em um envelope JSON validado por schema para consumo posterior por um gerador de laudo (LLM).

> **Aviso clínico:** esta é uma ferramenta de apoio à decisão para fins acadêmicos, NÃO um instrumento diagnóstico. Toda saída deve ser revisada por um oftalmologista antes de qualquer uso clínico.

---

## Estrutura do repositório

```
oct-image-analyzer/
├── pai-analise_retina-rafael_tardelli.ipynb   # Notebook principal (15 seções)
├── schema.json                                # JSON Schema draft-07 do envelope de saída (schema_version "1.0")
├── requirements.txt                           # Dependências fixadas (pinned)
├── .env.example                               # Variáveis de configuração (versionado)
├── .env                                       # Valores locais (NÃO versionado)
├── .gitignore                                 # Exclui .env, *.pth, outputs/*, *.npy, .venv/, caches, Zone.Identifier
├── oct_exemplo.jpg                            # Imagem OCT de exemplo (fallback de entrada)
├── data/
│   └── .gitkeep                               # Imagens opcionais para modo batch (oct_* / fundus_*)
├── outputs/
│   └── .gitkeep                               # JSONs gerados pelo notebook
└── old/                                       # Notebooks anteriores de modelo único (histórico)
```

### Seções do notebook

O notebook possui 15 seções numeradas: identificação, descrição do problema, base de dados, metodologia, setup, carregamento dos modelos, entrada e detecção de modalidade, pré-processamento, inferência e roteamento, pós-processamento / confiança / recomendações, geração e validação do JSON, stub de laudo, experimentos, análise e conclusões, limitações.

---

## Modelos

Ambos são ViT-Large/16 fine-tuned a partir do encoder RETFound. Carregados via `timm`, licença **CC-BY-NC-4.0 (uso não-comercial)**. Os repositórios são públicos — nenhuma autenticação no Hugging Face é necessária.

| # | Repositório HF | Modalidade | Classes | Dataset de fine-tune | Acurácia reportada |
|---|---|---|---|---|---|
| A | `bitfount/RETFound_MAE_OCT_CNV_DME_DRU` | OCT B-scan | CNV, DME, DRUSEN, NORMAL (4) | Kermany/OCT2017 | ~85% |
| B | `bitfount/RETFound_DR_IDRID` | Retinografia colorida | No DR, Mild DR, Moderate DR, Severe DR, Progressed DR — graus ICDR 0–4 (5) | IDRiD | — |

**Notas importantes:**

- O modelo A é carregado com `timm.create_model("hf_hub:bitfount/RETFound_MAE_OCT_CNV_DME_DRU", pretrained=True)`.
- O modelo B é carregado com `timm.create_model("hf_hub:bitfount/RETFound_DR_IDRID", pretrained=True)`.
- A biblioteca `transformers` **não é utilizada**. O pré-processamento é resolvido via `timm.data.resolve_data_config` + `create_transform`: 224×224, bicúbico, center crop 0.9, mean/std ImageNet.
- O modelo B foi treinado sobre imagens em **escala de cinza**. O notebook converte retinografias coloridas para greyscale antes da inferência.
- Os checkpoints são baixados e cacheados automaticamente pelo `timm` no primeiro uso. Não há download manual.

### Ordem dos rótulos (labels_source)

- **Modelo A:** `labels_source = "manual"`. A ordem `["CNV", "DME", "DRUSEN", "NORMAL"]` é **inferida** pela convenção alfabética `ImageFolder` do Kermany/OCT2017 e pelo sufixo do nome do repositório — **não está documentada pelo autor do checkpoint**. Deve ser validada com imagens de classe conhecida antes de confiar nos rótulos de classificação.
- **Modelo B:** `labels_source = "manual"`. A ordem dos graus ICDR 0–4 é explícita no model card.

---

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / WSL / macOS
# Windows: .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name pai-retina --display-name "Python (PAI Retina)"
cp .env.example .env               # edite se necessário
```

Abra o notebook e selecione o kernel **"Python (PAI Retina)"**. Os pesos dos modelos são baixados automaticamente do Hugging Face Hub na primeira execução.

---

## Variáveis de configuração (`.env`)

Copie `.env.example` para `.env` e ajuste conforme necessário. Todos os valores abaixo são os defaults.

| Variável | Default | Descrição |
|---|---|---|
| `MODEL_OCT_CHECKPOINT` | `bitfount/RETFound_MAE_OCT_CNV_DME_DRU` | Repositório HF do modelo OCT |
| `MODEL_FUNDUS_CHECKPOINT` | `bitfount/RETFound_DR_IDRID` | Repositório HF do modelo de retinografia |
| `DATA_DIR` | `./data` | Diretório de entrada para modo batch |
| `OUTPUT_DIR` | `./outputs` | Diretório de saída dos JSONs |
| `DEVICE_PREFERENCE` | `auto` | Dispositivo: `auto` (CUDA→MPS→CPU), `cuda`, `mps` ou `cpu` |
| `SEED` | `42` | Semente para reprodutibilidade |
| `CONF_PMAX_HIGH` | `0.85` | Limiar de p_max para confiança alta |
| `CONF_MARGIN_HIGH` | `0.30` | Limiar de margem (top1 − top2) para confiança alta |
| `CONF_PMAX_LOW` | `0.60` | Limiar de p_max abaixo do qual a confiança é baixa |
| `CONF_ENTROPY_HIGH` | `0.75` | Limiar de entropia normalizada para confiança baixa |
| `MODALITY_SATURATION_THRESHOLD` | `0.12` | Saturação média HSV que separa OCT (baixa) de retinografia (alta) |

---

## Modos de entrada

**Interativo (padrão):** dois slots `ipywidgets.FileUpload` — "Imagem de OCT (B-scan)" e "Retinografia colorida (fundo de olho)". A modalidade é declarada pelo slot utilizado.

**Batch (opcional):** deposite imagens em `data/` seguindo a convenção de nomes: `oct_*` para OCT e `fundus_*` para retinografia colorida. O notebook usa até duas imagens por exame (uma de cada modalidade).

**Fallback:** se nenhum upload for realizado e `data/` estiver vazio, o notebook utiliza `oct_exemplo.jpg` declarado como OCT, garantindo execução completa em um runtime limpo.

**Guarda de modalidade:** a saturação média no espaço HSV distingue imagens OCT (quase acromáticas, saturação baixa) de retinografias (alta saturação). Se a modalidade estimada divergir da declarada, o modelo incompatível **não é executado** e fica registrado com `status: "skipped_modality_mismatch"` e o flag `modality_mismatch_warning` em `quality_flags`.

---

## Envelope JSON de saída (`schema_version: "1.0"`)

Cada execução grava um arquivo JSON em `outputs/`, validado contra `schema.json` (JSON Schema draft-07) com a biblioteca `jsonschema`.

### Chaves de nível superior

| Chave | Tipo | Descrição |
|---|---|---|
| `exam_id` | string (UUID) | Identificador único do exame |
| `timestamp_utc` | string (ISO-8601) | Momento da geração |
| `schema_version` | string | `"1.0"` |
| `inputs` | array | Uma entrada por imagem fornecida |
| `models` | array | Um item por modelo (OCT e FUNDUS) |
| `recommendations` | array de strings | Recomendações determinísticas |
| `disclaimer` | string | Aviso de uso não clínico |

### Cada item de `inputs`

`input_id`, `filename`, `sha256`, `original_size` [w, h], `modality_declared` (OCT|FUNDUS), `modality_estimated` (OCT|FUNDUS|undetermined), `modality_agreement` (bool), `saturation_mean`, `preprocessing` (lista de etapas aplicadas), `quality_flags` (lista de flags de qualidade).

### Cada item de `models`

| Campo | Descrição |
|---|---|
| `name` | Nome identificador do modelo |
| `checkpoint` | Repositório HF |
| `task` | Descrição da tarefa |
| `modality_expected` | OCT ou FUNDUS |
| `input_id` | Referência ao input associado (ou `null`) |
| `status` | `ok` \| `no_input` \| `skipped_modality_mismatch` \| `error` |
| `labels_source` | `checkpoint` ou `manual` |
| `labels` | Lista de rótulos das classes |
| `probabilities` | Mapa `{label: probabilidade}` em [0, 1] |
| `top1` | `{label, probability}` — classe predita |
| `confidence` | `{p_max, margin, entropy_norm, level}` — ver seção abaixo |
| `inference_time_ms` | Tempo de inferência em ms |
| `error` | Mensagem de erro (quando aplicável) |

Os campos de predição (`labels`, `probabilities`, `top1`, `confidence`, `inference_time_ms`) são `null` sempre que `status != "ok"` — nunca zeros nem distribuição uniforme. As probabilidades de um modelo com status `ok` somam ≈1 (verificação extra feita no notebook, além da validação de schema).

### Níveis de confiança (determinísticos)

| Nível | Critério |
|---|---|
| `alta` | p_max ≥ 0.85 **E** margin ≥ 0.30 |
| `baixa` | p_max < 0.60 **OU** entropy_norm > 0.75 |
| `média` | demais casos |

### Recomendações (regras determinísticas, sem LLM)

- Confiança baixa → repetir captura ou buscar segunda opinião.
- Achado patológico em OCT com confiança média → acompanhamento mais próximo.
- Grau ICDR ≥ 3 → encaminhamento prioritário.
- Apenas uma modalidade avaliada → avaliação parcial do exame.
- Imagem de baixa qualidade → repetir captura.

---

## Stub de laudo

A função `gerar_laudo(json_saida: dict, layout: str) -> str` está declarada no notebook mas levanta `NotImplementedError` — o template de laudo será fornecido em etapa posterior do projeto. Quando implementado, o LLM deve receber apenas campos interpretáveis (`inputs`, `models.labels` / `probabilities` / `top1` / `confidence`, `recommendations`). Embeddings brutos **nunca** devem ser enviados ao LLM.

---

## Limitações

- **Cobertura restrita:** apenas 4 categorias OCT e 5 graus de RD; achados fora dessas classes não são detectados.
- **Generalização limitada:** o Kermany/OCT2017 é de único equipamento; o IDRiD possui 516 imagens de um único centro e população — a generalização para outros contextos clínicos não está estabelecida.
- **Ordem de rótulos do Modelo A não verificada:** deve ser validada com imagens de classe conhecida do Kermany/OCT2017 antes de qualquer uso.
- **Heurística de modalidade simplificada:** utiliza um único limiar empírico de saturação HSV, que pode falhar em imagens atípicas.
- **Licença não-comercial:** CC-BY-NC-4.0. Adequada para estudo e pesquisa; uso comercial requer verificação de licença.
- **Ferramenta acadêmica de apoio à decisão:** toda saída requer revisão por oftalmologista, conforme o `disclaimer` fixo presente em cada envelope JSON.

---

## Licença dos modelos

`bitfount/RETFound_MAE_OCT_CNV_DME_DRU` e `bitfount/RETFound_DR_IDRID` — **CC-BY-NC-4.0** (não-comercial).
