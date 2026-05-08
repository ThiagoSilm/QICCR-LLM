# QICCR-LLM v5.4

> Transformer de linguagem implementado do zero em Python puro — sem PyTorch, sem NumPy, sem frameworks.

---

## O que é isso

QICCR é um modelo de linguagem transformer completo escrito inteiramente com a biblioteca padrão do Python. Cada operação — forward pass, backward pass, otimizador Adam, KV cache, embeddings rotacionais — é implementada manualmente com `array.array('f')` e loops explícitos.

É um sistema educacional e experimental. Não compete com modelos de produção em velocidade, mas é matematicamente correto e totalmente inspecionável.

**Arquitetura resumida:**

| Parâmetro | Valor |
|---|---|
| Camadas | 2 |
| Cabeças de atenção | 4 |
| Dimensão do modelo | 128 |
| Dimensão FFN | 256 |
| Tamanho do vocab | 8192 |
| Contexto máximo (treino) | 64 tokens |
| KV cache máximo | 128 tokens |

---

## Requisitos

- Python 3.8 ou superior
- Nenhuma dependência externa

```bash
python --version  # 3.8+
```

---

## Instalação

```bash
git clone <repo>
cd qiccr
# pronto — não há pip install
```

---

## Uso rápido

### Treinar o modelo

Crie um arquivo `treino.txt` com o texto de treinamento (quanto maior e mais rico, melhor):

```bash
python qiccr_v54.py --train
```

Com número de épocas personalizado:

```bash
python qiccr_v54.py --train 10
```

O treinamento salva checkpoints automaticamente:
- `qiccr_v5_best_*` — melhor loss vista até agora
- `qiccr_v5_latest_*` — checkpoint da última época

### Conversar com o modelo

```bash
python qiccr_v54.py
```

O modelo carrega o melhor checkpoint disponível e abre o chat interativo:

```
🚀 QICCR v5.4 | 'sair' | 'reset'

🧑 Você: olá, como vai?
🐶 Qiccr: ...
```

**Comandos especiais no chat:**

| Comando | Ação |
|---|---|
| `sair` / `exit` / `quit` | Encerra o programa |
| `reset` | Limpa o KV cache e reinicia o contexto |

---

## Configuração

Todos os hiperparâmetros ficam na classe `Config` no topo do arquivo. Edite diretamente antes de treinar:

```python
class Config:
    VOCAB_SIZE = 8192        # tamanho do vocabulário BPE
    D_MODEL = 128            # dimensão dos embeddings
    N_HEADS = 4              # cabeças de atenção multi-head
    N_LAYERS = 2             # número de blocos transformer
    D_FF = 256               # dimensão interna do FFN
    MAX_SEQ = 64             # janela de contexto no treino
    KV_MAX_SEQ = 128         # slots máximos no KV cache
    LEARNING_RATE = 3e-4     # taxa de aprendizado AdamW
    WEIGHT_DECAY = 0.01      # regularização L2
    GRAD_CLIP = 1.0          # clipping de gradiente
    FFN_DROPOUT = 0.1        # dropout no FFN (apenas treino)
    TEMPERATURE = 0.75       # temperatura de amostragem
    TOP_K = 50               # top-k na geração
    TOP_P = 0.90             # top-p (nucleus sampling)
    REPETITION_PENALTY = 1.1 # penalidade por repetição
    MAX_TRAIN_STEPS = 8000   # passos por época
    TRAIN_WINDOW_SIZE = 32   # janela deslizante de treino
```

**Dicas de configuração:**

- Para textos curtos (~10k chars): reduza `MAX_TRAIN_STEPS` para 2000–3000
- Para aumentar capacidade: aumente `D_MODEL` para 256 e `N_LAYERS` para 4 (treino ~4x mais lento)
- Para geração mais criativa: `TEMPERATURE = 0.9`, `TOP_P = 0.95`
- Para geração mais determinística: `TEMPERATURE = 0.5`, `TOP_K = 20`

---

## Uso programático

```python
from qiccr_v54 import QICCRLLM, Config
import random

random.seed(42)

# Instanciar
model = QICCRLLM()

# Treinar
from qiccr_v54 import train_model
train_model(model, filepath="meu_texto.txt", epochs=3)

# Gerar texto
resposta = model.generate(
    "Era uma vez",
    max_new=100,
    temperature=0.8,
    top_k=50,
    top_p=0.92
)
print(resposta)

# Salvar e carregar
model.save("meu_modelo")
model.load("meu_modelo")
```

### Tokenizar manualmente

```python
tokens = model.tokenizer.encode("olá mundo")
print(tokens)  # tuple de inteiros

texto = model.tokenizer.decode(list(tokens))
print(texto)   # "olá mundo"
```

### Treinar o tokenizador separadamente

O tokenizador BPE pode ser treinado em qualquer lista de textos:

```python
textos = ["primeiro texto...", "segundo texto..."]
model.tokenizer.train(textos, num_merges=3000)
```

---

## Como funciona

### Fluxo geral

```
Texto → BPE Tokenizer → Token IDs
Token IDs → Embedding → Vetores D_MODEL
Vetores → [TransformerBlock × N_LAYERS] → Representações
Representação final → LayerNorm → Logits
Logits → Softmax → Probabilidades → Token gerado
```

### 1. Tokenizador BPE

O tokenizador usa Byte Pair Encoding com um heap de prioridade para aplicar merges em O(n log n). Parte de um vocabulário base de todos os 256 bytes, aprende pares frequentes no texto de treinamento, e constrói tokens progressivamente maiores.

Tokens especiais:
- `<pad>` (0) — preenchimento
- `<unk>` (1) — token desconhecido
- `<s>` (2) — início de sequência
- `</s>` (3) — fim de sequência

### 2. Armazenamento de pesos

Todos os parâmetros vivem em um único `array.array('f')` flat. Cada camada recebe um offset calculado pelo `WeightAllocator` na inicialização. Isso simula o comportamento de um tensor store sem usar NumPy.

```
[tok_embed | q0 k0 v0 o0 up0 down0 ln10 ln20 | q1 k1 ... | ln_final | logit_scale]
 ←── VOCAB*D ──→ ←──────────── LAYER 0 ────────────────→ ←─ LAYER 1 ─→
```

### 3. Atenção multi-head com RoPE

Cada token projeta Q, K, V. As queries e keys recebem embeddings rotacionais posicionais (RoPE) que codificam posição via rotações no espaço complexo:

```
q_rotated[2i]   = q[2i]   * cos(θ_i) - q[2i+1] * sin(θ_i)
q_rotated[2i+1] = q[2i]   * sin(θ_i) + q[2i+1] * cos(θ_i)
θ_i = pos / 10000^(2i/d_head)
```

O RoPE permite que o modelo aprenda relações relativas de posição, não absolutas.

### 4. KV Cache

Durante a inferência, as chaves e valores já computados são armazenados e reutilizados. Cada entrada tem:
- `slot_id` — identificador global único
- `timestamp` — posição original na sequência
- `K_base` — chave **sem** rotação aplicada (rotacionada no momento do uso)

Isso garante que a rotação RoPE use sempre o timestamp correto, independente da ordem de armazenamento no cache.

### 5. Backward pass manual

O gradiente flui explicitamente de volta por cada operação:

```
∂L/∂logits → ∂L/∂final → ∂L/∂LayerNorm → ∂L/∂TransformerBlocks → ∂L/∂embeddings
```

Dentro de cada bloco, o backward percorre em ordem inversa:
1. FFN down → GELU' → FFN up
2. LayerNorm 2 + residual
3. Output projection
4. Atenção: ∂L/∂probs → softmax backward → ∂L/∂scores → ∂L/∂Q, ∂K, ∂V
5. Inverse RoPE nos gradientes de Q e K
6. Projeções Q, K, V
7. LayerNorm 1 + residual

### 6. Otimizador AdamW

Implementação completa com:
- Momentos de primeira e segunda ordem (m, v)
- Correção de bias: `m̂ = m / (1 - β₁ᵗ)`
- Weight decay desacoplado (AdamW, não Adam)
- Gradient clipping global por norma L2

### 7. Geração de texto

A geração usa o pipeline prefill + decode:

1. **Prefill**: processa o prompt inteiro, popula o KV cache
2. **Decode**: gera um token por vez, cada passo usa o KV cache acumulado

Amostragem com:
- Top-K: mantém apenas os K tokens mais prováveis
- Top-P (nucleus): mantém os tokens que somam probabilidade ≥ P
- Penalidade de repetição: divide logits de tokens recentes por 1.1

---

## Checkpoints

Três arquivos são salvos por checkpoint:

| Arquivo | Conteúdo |
|---|---|
| `*_fp32.bin` | Pesos do modelo em float32 raw |
| `*_optim.json.gz` | Estado do Adam (m, v, t) comprimido |
| `*_meta.json.gz` | Vocabulário BPE, merges, step_count |

Para carregar manualmente:

```python
model = QICCRLLM()
model.load("qiccr_v5_best")   # sem extensão
```

---

## Expectativas de desempenho

Este modelo é **Python puro** — sem vetorização, sem GPU. Tempos aproximados em CPU moderna:

| Operação | Tempo estimado |
|---|---|
| 1 passo de treino (janela 32) | 5–15 segundos |
| Gerar 1 token | 2–5 segundos |
| Epoch de 500 passos | ~1–2 horas |

Para uso prático de treinamento, recomenda-se textos de até ~50k caracteres e 1–3 épocas.

A convergência é real — a loss decresce — mas o tempo por passo limita experimentos extensos.

---

## Arquitetura de arquivos

```
qiccr_v54.py           # código principal — tudo em um arquivo
treino.txt             # seu corpus de treinamento
qiccr_v5_best_fp32.bin        # pesos do melhor checkpoint
qiccr_v5_best_optim.json.gz   # estado do otimizador
qiccr_v5_best_meta.json.gz    # vocab e metadados
qiccr_v5_latest_*             # checkpoint da última época
```

---

## Limitações conhecidas

**Performance:** O gargalo principal é o produto interno do lm_head: VOCAB_SIZE × D_MODEL operações em Python puro por forward pass (~1M operações). Não há caminho para GPU nesta implementação.

**KV cache no treino:** O KV cache é usado apenas na inferência. O treino sempre computa atenção completa na janela.

**Tamanho do modelo:** Com os hiperparâmetros padrão, o modelo tem ~1.3M parâmetros — suficiente para memorizar padrões locais em textos curtos, mas limitado para generalização.

---

## Histórico de versões

| Versão | Mudanças principais |
|---|---|
| v5.4 | Fix `inverse_rotate` com timestamp correto; `encode.cache_clear()` pós-treino BPE; `random.seed(42)` |
| v5.3 | KV cache com `slot_ids` globais; `safe_softmax` com fallback uniforme; `logit_scale` com clamp |

---

## Licença

Uso livre para fins educacionais e de pesquisa.
