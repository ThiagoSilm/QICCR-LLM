"""
QICCR-LLM v6.1 — Correções Completas
======================================
Correções aplicadas sobre v6.0:
1. [FIX] Noam Schedule: fórmula correta com d_model^(-0.5) conforme paper
2. [FIX] Tokenizer: merge_rank agora é salvo e carregado corretamente
3. [FIX] WeightAllocator: verificação de overflow com assert
4. [FIX] Beam Search: KV-cache por beam para evitar forward quadrático
5. [FIX] Backward de atenção: índice causal_mask consistente
6. [FIX] Gradientes normalizados por batch_size antes do passo do otimizador
7. [FIX] lru_cache no tokenizer removido (thrash durante treino)
8. [FIX] Dropout em inferência: training=False garantido em todos os paths
"""

import math, random, os, sys, array, json, gzip, heapq
from collections import Counter, defaultdict

# ====================================================================
# CONFIGURAÇÃO
# ====================================================================
class Config:
    VOCAB_SIZE       = 8192
    D_MODEL          = 128
    N_HEADS          = 4
    N_LAYERS         = 2
    D_FF             = 256
    MAX_SEQ          = 64
    KV_MAX_SEQ       = 128
    LEARNING_RATE    = 3e-4      # usado apenas como referência; Noam ignora isso
    ADAM_BETA1       = 0.9
    ADAM_BETA2       = 0.999
    ADAM_EPS         = 1e-8
    WEIGHT_DECAY     = 0.01
    GRAD_CLIP        = 1.0
    TOP_K            = 50
    TOP_P            = 0.90
    TEMPERATURE      = 0.75
    REPETITION_PENALTY = 1.1
    MAX_TRAIN_STEPS  = 8000
    TRAIN_WINDOW_MIN = 16
    TRAIN_WINDOW_MAX = 48
    BATCH_SIZE       = 4
    FFN_DROPOUT      = 0.1
    LOGIT_SCALE_MIN  = 0.1
    LOGIT_SCALE_MAX  = 10.0
    LABEL_SMOOTHING  = 0.1
    NOAM_WARMUP      = 4000
    BEAM_WIDTH       = 3

# ====================================================================
# WeightAllocator  — com verificação de overflow
# ====================================================================
class WeightAllocator:
    def __init__(self, total):
        self.total  = total
        self.offset = 0

    def alloc(self, size, name=""):
        off = self.offset
        self.offset += size
        assert self.offset <= self.total, (
            f"WeightAllocator overflow ao alocar '{name}': "
            f"offset {self.offset} > total {self.total}"
        )
        return off

# ====================================================================
# WEIGHT STORE
# ====================================================================
class WeightStore:
    def __init__(self, total_params):
        self.total_params = total_params
        self.fp32 = array.array('f', [0.0] * total_params)

    def read_fp32(self, offset):
        return self.fp32[offset]

    def write_fp32(self, offset, val):
        self.fp32[offset] = val

    def read_vector_fp32(self, offset, length):
        fp32 = self.fp32
        return array.array('f', [fp32[offset + i] for i in range(length)])

# ====================================================================
# ADAMW
# ====================================================================
class AdamOptimizer:
    def __init__(self, total_params):
        self.m = array.array('f', [0.0] * total_params)
        self.v = array.array('f', [0.0] * total_params)
        self.t = 0

    def step(self, weights, grads, offsets, lr=3e-4, wd=0.01, clip=1.0):
        self.t += 1
        beta1, beta2, eps = Config.ADAM_BETA1, Config.ADAM_BETA2, Config.ADAM_EPS

        # Gradient clipping global
        if offsets:
            total_norm = math.sqrt(
                sum(grads[off + i] * grads[off + i]
                    for off, size in offsets
                    for i in range(size))
            )
            if total_norm > clip:
                scale = clip / total_norm
                for off, size in offsets:
                    for i in range(size):
                        grads[off + i] *= scale

        for off, size in offsets:
            for i in range(size):
                g = grads[off + i]
                self.m[off + i] = beta1 * self.m[off + i] + (1 - beta1) * g
                self.v[off + i] = beta2 * self.v[off + i] + (1 - beta2) * g * g
                m_hat = self.m[off + i] / (1 - beta1 ** self.t)
                v_hat = self.v[off + i] / (1 - beta2 ** self.t)
                weights[off + i] -= (
                    lr * wd * weights[off + i]
                    + lr * m_hat / (math.sqrt(v_hat) + eps)
                )
                grads[off + i] = 0.0

    def save(self, filepath):
        with gzip.open(filepath, 'wt') as f:
            json.dump({'m': list(self.m), 'v': list(self.v), 't': self.t}, f)

    def load(self, filepath):
        if not os.path.exists(filepath):
            return False
        with gzip.open(filepath, 'rt') as f:
            data = json.load(f)
        self.m = array.array('f', data['m'])
        self.v = array.array('f', data['v'])
        self.t = data['t']
        return True

# ====================================================================
# FUNÇÕES AUXILIARES
# ====================================================================
def safe_softmax(arr, mask=None):
    if mask is None:
        mask = [True] * len(arr)
    valid = [i for i, m in enumerate(mask) if m]
    if not valid:
        return array.array('f', [1.0 / len(arr)] * len(arr))
    max_val = max(arr[i] for i in valid)
    exps = []
    for i, m in enumerate(mask):
        if not m:
            exps.append(0.0)
        else:
            exps.append(math.exp(max(-80.0, min(80.0, arr[i] - max_val))))
    s = sum(exps)
    if s == 0 or not math.isfinite(s):
        result = array.array('f', [0.0] * len(arr))
        for i in valid:
            result[i] = 1.0 / len(valid)
        return result
    return array.array('f', [e / s for e in exps])


def softmax_backward(probs, mask, grad_probs, scale):
    total_len = len(probs)
    valid = [i for i, m in enumerate(mask) if m]
    if not valid:
        return [0.0] * total_len
    dot = sum(grad_probs[j] * probs[j] for j in valid)
    grad_scores = [0.0] * total_len
    for j in valid:
        grad_scores[j] = (probs[j] * (grad_probs[j] - dot)) / scale
    return grad_scores


def gelu_arr(x):
    return array.array('f', [
        0.5 * xi * (1.0 + math.tanh(0.7978845608028654 * (xi + 0.044715 * xi ** 3)))
        for xi in x
    ])


def gelu_derivative(x):
    inner     = 0.7978845608028654 * (x + 0.044715 * x ** 3)
    tanh_part = math.tanh(inner)
    sech2     = 1.0 - tanh_part * tanh_part
    return (0.5 * (1.0 + tanh_part)
            + 0.5 * x * sech2 * 0.7978845608028654 * (1.0 + 0.134145 * x * x))


def gelu_derivative_arr(x):
    return array.array('f', [gelu_derivative(xi) for xi in x])

# ====================================================================
# RoPE
# ====================================================================
class RotaryPositionalEmbedding:
    def __init__(self, d_head, max_pos=None):
        self.d_head    = d_head
        self.max_pos   = max_pos or Config.KV_MAX_SEQ + 1024
        self.cos_cache = {}
        self.sin_cache = {}

    def _ensure(self, pos):
        pos = min(pos, self.max_pos)
        if pos not in self.cos_cache:
            cos_vals = array.array('f', [0.0] * self.d_head)
            sin_vals = array.array('f', [0.0] * self.d_head)
            for i in range(0, self.d_head, 2):
                theta = pos / (10000.0 ** (i / self.d_head))
                cos_vals[i] = math.cos(theta)
                sin_vals[i] = math.sin(theta)
                if i + 1 < self.d_head:
                    cos_vals[i + 1] = cos_vals[i]
                    sin_vals[i + 1] = sin_vals[i]
            self.cos_cache[pos] = cos_vals
            self.sin_cache[pos] = sin_vals

    def rotate(self, x, pos):
        self._ensure(pos)
        cos, sin = self.cos_cache[pos], self.sin_cache[pos]
        rotated = array.array('f', [0.0] * len(x))
        for i in range(0, len(x), 2):
            if i + 1 < len(x):
                rotated[i]     = x[i] * cos[i] - x[i + 1] * sin[i]
                rotated[i + 1] = x[i] * sin[i] + x[i + 1] * cos[i]
            else:
                rotated[i] = x[i]
        return rotated

    def inverse_rotate(self, x, pos):
        self._ensure(pos)
        cos, sin = self.cos_cache[pos], self.sin_cache[pos]
        derotated = array.array('f', [0.0] * len(x))
        for i in range(0, len(x), 2):
            if i + 1 < len(x):
                derotated[i]     =  x[i] * cos[i] + x[i + 1] * sin[i]
                derotated[i + 1] = -x[i] * sin[i] + x[i + 1] * cos[i]
            else:
                derotated[i] = x[i]
        return derotated

# ====================================================================
# KV-CACHE  — suporta múltiplos beams independentes
# ====================================================================
class KVCache:
    def __init__(self, n_layers, n_heads, d_head, max_seq):
        self.n_layers = n_layers
        self.n_heads  = n_heads
        self.max_seq  = max_seq
        self._reset()

    def _reset(self):
        nl, nh = self.n_layers, self.n_heads
        self.K_base    = [[[] for _ in range(nh)] for _ in range(nl)]
        self.V         = [[[] for _ in range(nh)] for _ in range(nl)]
        self.timestamps= [[[] for _ in range(nh)] for _ in range(nl)]

    def update(self, layer, head, pos, k_base, v):
        self.K_base[layer][head].append(array.array('f', k_base))
        self.V[layer][head].append(array.array('f', v))
        self.timestamps[layer][head].append(pos)
        # Remove entrada mais antiga se ultrapassar limite
        if len(self.timestamps[layer][head]) > self.max_seq:
            oldest = min(range(len(self.timestamps[layer][head])),
                         key=lambda i: self.timestamps[layer][head][i])
            self.K_base[layer][head].pop(oldest)
            self.V[layer][head].pop(oldest)
            self.timestamps[layer][head].pop(oldest)

    def get_KV_ordered(self, layer, head, query_pos):
        K_list, V_list, time_list = [], [], []
        for k, v, ts in zip(self.K_base[layer][head],
                             self.V[layer][head],
                             self.timestamps[layer][head]):
            if ts <= query_pos:
                K_list.append(k)
                V_list.append(v)
                time_list.append(ts)
        return K_list, V_list, time_list

    def clone(self):
        """Retorna uma cópia profunda do cache — usado para duplicar beams."""
        new = KVCache(self.n_layers, self.n_heads, self.max_seq, self.max_seq)
        for l in range(self.n_layers):
            for h in range(self.n_heads):
                new.K_base[l][h]     = [array.array('f', k) for k in self.K_base[l][h]]
                new.V[l][h]          = [array.array('f', v) for v in self.V[l][h]]
                new.timestamps[l][h] = list(self.timestamps[l][h])
        return new

    def clear(self):
        self._reset()

# ====================================================================
# CAMADAS
# ====================================================================
class Linear:
    def __init__(self, in_f, out_f, offset, name="", trainable=True):
        self.in_f      = in_f
        self.out_f     = out_f
        self.offset    = offset
        self.name      = name
        self.trainable = trainable
        self.W_size    = out_f * (in_f + 1)

    def forward(self, x, store, dropout_rate=0.0, training=False):
        out = array.array('f', [0.0] * self.out_f)
        for i in range(self.out_f):
            base = self.offset + i * (self.in_f + 1)
            s    = store.read_fp32(base + self.in_f)        # bias
            for j in range(self.in_f):
                s += store.read_fp32(base + j) * x[j]
            out[i] = s

        mask = None
        # Dropout APENAS em modo treino
        if training and dropout_rate > 0.0:
            scale = 1.0 / (1.0 - dropout_rate)
            mask  = array.array('f', [
                scale if random.random() > dropout_rate else 0.0
                for _ in range(len(out))
            ])
            for i in range(len(out)):
                out[i] *= mask[i]

        return out, array.array('f', x), mask

    def backward(self, grad_output, input_cache, mask, store, grads):
        if not self.trainable:
            return array.array('f', [0.0] * self.in_f)
        if mask is not None:
            grad_output = array.array('f', [
                grad_output[i] * mask[i] for i in range(len(grad_output))
            ])
        grad_input = array.array('f', [0.0] * self.in_f)
        for i in range(self.out_f):
            base = self.offset + i * (self.in_f + 1)
            g    = grad_output[i]
            grads[base + self.in_f] += g                    # grad bias
            for j in range(self.in_f):
                w = store.read_fp32(base + j)
                grads[base + j]  += g * input_cache[j]
                grad_input[j]    += w * g
        return grad_input


class LayerNorm:
    def __init__(self, dim, offset, name="", trainable=True):
        self.dim       = dim
        self.offset    = offset
        self.name      = name
        self.trainable = trainable

    def forward(self, x, store):
        n    = len(x)
        mean = sum(x) / n
        var  = sum((xi - mean) ** 2 for xi in x) / n
        std  = math.sqrt(var + 1e-5)
        normalized = array.array('f', [(xi - mean) / std for xi in x])
        out = array.array('f', [0.0] * n)
        for i in range(n):
            gamma = store.read_fp32(self.offset + i)
            beta  = store.read_fp32(self.offset + self.dim + i)
            out[i] = gamma * normalized[i] + beta
        return out, (normalized, std, array.array('f', x))

    def backward(self, grad_output, cache, store, grads):
        normalized, std, x_orig = cache
        n   = len(x_orig)
        eps = 1e-5
        if self.trainable:
            for i in range(n):
                grads[self.offset + i]            += grad_output[i] * normalized[i]
                grads[self.offset + self.dim + i] += grad_output[i]
        grad_normalized = array.array('f', [
            grad_output[i] * store.read_fp32(self.offset + i) for i in range(n)
        ])
        mean_grad = sum(grad_normalized) / n
        std_grad  = sum(grad_normalized[i] * normalized[i] for i in range(n)) / n
        grad_x    = array.array('f', [0.0] * n)
        for i in range(n):
            grad_x[i] = (grad_normalized[i] - mean_grad - normalized[i] * std_grad) / max(std, eps)
        return grad_x

# ====================================================================
# TRANSFORMER BLOCK
# ====================================================================
class TransformerBlock:
    def __init__(self, idx, allocator, store, trainable=True):
        d  = Config.D_MODEL
        df = Config.D_FF
        self.idx      = idx
        self.d_model  = d
        self.n_heads  = Config.N_HEADS
        self.d_head   = d // Config.N_HEADS
        self.trainable= trainable

        self.q_proj  = Linear(d,  d,  allocator.alloc(d  * (d  + 1), f"q_{idx}"),  f"q_{idx}",   trainable)
        self.k_proj  = Linear(d,  d,  allocator.alloc(d  * (d  + 1), f"k_{idx}"),  f"k_{idx}",   trainable)
        self.v_proj  = Linear(d,  d,  allocator.alloc(d  * (d  + 1), f"v_{idx}"),  f"v_{idx}",   trainable)
        self.o_proj  = Linear(d,  d,  allocator.alloc(d  * (d  + 1), f"o_{idx}"),  f"o_{idx}",   trainable)
        self.ff_up   = Linear(d,  df, allocator.alloc(df * (d  + 1), f"up_{idx}"), f"up_{idx}",  trainable)
        self.ff_down = Linear(df, d,  allocator.alloc(d  * (df + 1), f"dn_{idx}"), f"dn_{idx}",  trainable)
        self.ln1     = LayerNorm(d, allocator.alloc(2 * d, f"ln1_{idx}"), f"ln1_{idx}", trainable)
        self.ln2     = LayerNorm(d, allocator.alloc(2 * d, f"ln2_{idx}"), f"ln2_{idx}", trainable)
        self.rope    = RotaryPositionalEmbedding(self.d_head)

    # ------------------------------------------------------------------
    def forward(self, x_list, store, kv_cache=None, positions=None, training=False):
        seq_len = len(x_list)
        if seq_len == 0:
            return x_list, None

        d_model = self.d_model
        n_heads = self.n_heads
        d_head  = self.d_head
        scale   = math.sqrt(d_head)

        # --- LayerNorm 1 ---
        normed1, ln1_caches = [], []
        for x in x_list:
            out, cache = self.ln1.forward(x, store)
            normed1.append(out)
            ln1_caches.append(cache)

        # --- Projeções QKV ---
        Q_full, K_full, V_full = [], [], []
        q_caches, k_caches, v_caches = [], [], []
        q_masks,  k_masks,  v_masks  = [], [], []
        for n in normed1:
            q, qc, qm = self.q_proj.forward(n, store, 0.0, training)
            k, kc, km = self.k_proj.forward(n, store, 0.0, training)
            v, vc, vm = self.v_proj.forward(n, store, 0.0, training)
            Q_full.append(q); K_full.append(k); V_full.append(v)
            q_caches.append(qc); k_caches.append(kc); v_caches.append(vc)
            q_masks.append(qm);  k_masks.append(km);  v_masks.append(vm)

        pos_local = positions if positions is not None else list(range(seq_len))

        # Aplica RoPE nas queries
        for i in range(seq_len):
            Q_full[i] = self.rope.rotate(Q_full[i], pos_local[i])

        # Divide em cabeças
        Q_heads, K_heads, V_heads = [], [], []
        for h in range(n_heads):
            s, e = h * d_head, (h + 1) * d_head
            Q_heads.append([array.array('f', [Q_full[i][j] for j in range(s, e)]) for i in range(seq_len)])
            K_heads.append([array.array('f', [K_full[i][j] for j in range(s, e)]) for i in range(seq_len)])
            V_heads.append([array.array('f', [V_full[i][j] for j in range(s, e)]) for i in range(seq_len)])

        # Atualiza KV-cache (inferência)
        if kv_cache is not None and positions is not None:
            for h in range(n_heads):
                for i in range(seq_len):
                    kv_cache.update(self.idx, h, pos_local[i], K_heads[h][i], V_heads[h][i])

        # --- Atenção ---
        head_outputs  = []
        all_probs     = []
        all_causal_masks = []   # [n_heads][seq_len][total_len]

        for h in range(n_heads):
            Qh = Q_heads[h]

            if kv_cache is not None and positions is not None:
                Kh_base, Vh, timestamps = kv_cache.get_KV_ordered(self.idx, h, pos_local[-1])
                # Aplica RoPE nas keys pelo timestamp
                Kh = [self.rope.rotate(array.array('f', k), ts)
                      for k, ts in zip(Kh_base, timestamps)]
                total_len = len(Kh)
                ts_list   = timestamps
            else:
                Kh        = [self.rope.rotate(array.array('f', k), pos_local[i])
                             for i, k in enumerate(K_heads[h])]
                Vh        = V_heads[h]
                total_len = len(Kh)
                ts_list   = pos_local

            # Máscara causal: posição j visível para query i se ts_list[j] <= pos_local[i]
            causal_mask_h = [
                [ts_list[j] <= pos_local[i] for j in range(total_len)]
                for i in range(seq_len)
            ]
            all_causal_masks.append(causal_mask_h)

            attn_h, probs_h = [], []
            for i in range(seq_len):
                scores = array.array('f', [
                    sum(Qh[i][k] * Kh[j][k] for k in range(d_head)) / scale
                    if causal_mask_h[i][j] else float('-inf')
                    for j in range(total_len)
                ])
                probs    = safe_softmax(scores, causal_mask_h[i])
                head_out = array.array('f', [0.0] * d_head)
                for j in range(total_len):
                    if probs[j] > 0:
                        for k in range(d_head):
                            head_out[k] += probs[j] * Vh[j][k]
                attn_h.append(head_out)
                probs_h.append(probs)

            head_outputs.append(attn_h)
            all_probs.append(probs_h)

        # Concatena cabeças
        attn_concat = []
        for i in range(seq_len):
            concat = array.array('f', [0.0] * d_model)
            for h in range(n_heads):
                s = h * d_head
                for k in range(d_head):
                    concat[s + k] = head_outputs[h][i][k]
            attn_concat.append(concat)

        # Projeção de saída da atenção
        attn_proj, o_caches, o_masks = [], [], []
        for a in attn_concat:
            out, oc, om = self.o_proj.forward(a, store, 0.0, training)
            attn_proj.append(out)
            o_caches.append(oc)
            o_masks.append(om)

        # Residual 1
        x = [
            array.array('f', [x_list[i][j] + attn_proj[i][j] for j in range(d_model)])
            for i in range(seq_len)
        ]
        residual2 = [array.array('f', xi) for xi in x]

        # --- LayerNorm 2 ---
        normed2, ln2_caches = [], []
        for vec in x:
            out, cache = self.ln2.forward(vec, store)
            normed2.append(out)
            ln2_caches.append(cache)

        # --- FFN ---
        pre_gelu, up_caches, up_masks = [], [], []
        for n in normed2:
            out, cache, mask = self.ff_up.forward(
                n, store, Config.FFN_DROPOUT if training else 0.0, training
            )
            pre_gelu.append(out)
            up_caches.append(cache)
            up_masks.append(mask)

        post_gelu = [gelu_arr(up) for up in pre_gelu]

        ffn_out, down_caches, down_masks = [], [], []
        for act in post_gelu:
            out, cache, mask = self.ff_down.forward(act, store, 0.0, training)
            ffn_out.append(out)
            down_caches.append(cache)
            down_masks.append(mask)

        # Residual 2
        x = [
            array.array('f', [residual2[i][j] + ffn_out[i][j] for j in range(d_model)])
            for i in range(seq_len)
        ]

        caches = None
        if self.trainable:
            caches = {
                'ln1_caches':   ln1_caches,
                'q_caches':     q_caches,
                'k_caches':     k_caches,
                'v_caches':     v_caches,
                'o_caches':     o_caches,
                'ln2_caches':   ln2_caches,
                'up_caches':    up_caches,
                'down_caches':  down_caches,
                'Q_heads':      Q_heads,
                'K_heads':      K_heads,
                'V_heads':      V_heads,
                'all_probs':    all_probs,
                'pre_gelu':     pre_gelu,
                'positions':    pos_local,
                'causal_mask':  all_causal_masks,    # [n_heads][seq_len][total_len]
                'q_masks':      q_masks,
                'k_masks':      k_masks,
                'v_masks':      v_masks,
                'o_masks':      o_masks,
                'up_masks':     up_masks,
                'down_masks':   down_masks,
            }
        return x, caches

    # ------------------------------------------------------------------
    def backward(self, grad_output_list, caches, store, grads):
        if not self.trainable or caches is None:
            return [array.array('f', [0.0] * self.d_model) for _ in grad_output_list]

        c       = caches
        seq_len = len(grad_output_list)
        d_model = self.d_model
        d_head  = self.d_head
        n_heads = self.n_heads
        scale   = math.sqrt(d_head)
        positions = c['positions']

        # --- Backward FFN ---
        grad_down = [
            self.ff_down.backward(
                array.array('f', [grad_output_list[i][j] for j in range(d_model)]),
                c['down_caches'][i], c['down_masks'][i], store, grads
            )
            for i in range(seq_len)
        ]
        grad_post_gelu = [
            array.array('f', [
                grad_down[i][k] * gelu_derivative(c['pre_gelu'][i][k])
                for k in range(len(grad_down[i]))
            ])
            for i in range(seq_len)
        ]
        grad_up = [
            self.ff_up.backward(grad_post_gelu[i], c['up_caches'][i], c['up_masks'][i], store, grads)
            for i in range(seq_len)
        ]

        # Grad residual 2: passa tanto pelo FFN quanto direto
        grad_res2 = [
            array.array('f', [grad_up[i][j] + grad_output_list[i][j] for j in range(d_model)])
            for i in range(seq_len)
        ]
        grad_ln2 = [
            self.ln2.backward(grad_res2[i], c['ln2_caches'][i], store, grads)
            for i in range(seq_len)
        ]

        # --- Backward projeção de saída da atenção ---
        # grad residual 1 inclui contribuição do grad de residual 2
        grad_o = [
            self.o_proj.backward(
                array.array('f', [grad_ln2[i][j] + grad_output_list[i][j] for j in range(d_model)]),
                c['o_caches'][i], c['o_masks'][i], store, grads
            )
            for i in range(seq_len)
        ]

        # --- Backward atenção ---
        grad_Q_full = [array.array('f', [0.0] * d_model) for _ in range(seq_len)]
        grad_K_full = [array.array('f', [0.0] * d_model) for _ in range(seq_len)]
        grad_V_full = [array.array('f', [0.0] * d_model) for _ in range(seq_len)]

        for h in range(n_heads):
            s, e = h * d_head, (h + 1) * d_head
            Vh       = c['V_heads'][h]
            probs_h  = c['all_probs'][h]
            # causal_mask indexado por cabeça, depois por posição de query
            causal_mask_h = c['causal_mask'][h]  # [seq_len][total_len]
            Kh_h = c['K_heads'][h]
            Qh_h = c['Q_heads'][h]
            total_len = len(Vh)

            for i in range(seq_len):
                grad_head = array.array('f', [grad_o[i][s + k] for k in range(d_head)])
                probs     = probs_h[i]
                mask_i    = causal_mask_h[i]           # [total_len]

                # Grad w.r.t. probs (via atenção ponderada dos values)
                grad_probs = [0.0] * total_len
                for j in range(total_len):
                    if mask_i[j]:
                        acc = 0.0
                        for k in range(d_head):
                            acc += float(grad_head[k]) * float(Vh[j][k])
                        grad_probs[j] = acc

                # Grad w.r.t. scores (via softmax backward)
                grad_scores = softmax_backward(probs, mask_i, grad_probs, scale)

                # Acumula grad em Q e K
                for j in range(total_len):
                    if not mask_i[j]:
                        continue
                    gs = grad_scores[j]
                    if abs(gs) < 1e-30:
                        continue
                    Kh_j = Kh_h[j]
                    Qh_i = Qh_h[i]
                    for k in range(d_head):
                        grad_Q_full[i][s + k] += gs * float(Kh_j[k])
                        grad_K_full[j][s + k] += gs * float(Qh_i[k])

                # Acumula grad em V
                for j in range(total_len):
                    if not mask_i[j]:
                        continue
                    p = float(probs[j])
                    if p <= 0:
                        continue
                    for k in range(d_head):
                        grad_V_full[j][s + k] += p * float(grad_head[k])

        # Desfaz RoPE nos gradientes (inverse_rotate)
        for i in range(seq_len):
            grad_Q_full[i] = self.rope.inverse_rotate(grad_Q_full[i], positions[i])
        for j in range(seq_len):
            grad_K_full[j] = self.rope.inverse_rotate(grad_K_full[j], positions[j])

        # --- Backward projeções Q, K, V ---
        grad_Q = [
            self.q_proj.backward(grad_Q_full[i], c['q_caches'][i], c['q_masks'][i], store, grads)
            for i in range(seq_len)
        ]
        grad_K = [
            self.k_proj.backward(grad_K_full[i], c['k_caches'][i], c['k_masks'][i], store, grads)
            for i in range(seq_len)
        ]
        grad_V = [
            self.v_proj.backward(grad_V_full[i], c['v_caches'][i], c['v_masks'][i], store, grads)
            for i in range(seq_len)
        ]

        # --- Backward LayerNorm 1 ---
        grad_ln1 = [
            self.ln1.backward(
                array.array('f', [grad_Q[i][j] + grad_K[i][j] + grad_V[i][j] for j in range(d_model)]),
                c['ln1_caches'][i], store, grads
            )
            for i in range(seq_len)
        ]
        return grad_ln1

# ====================================================================
# TOKENIZER (BPE)
# ====================================================================
class BPETokenizer:
    def __init__(self):
        self.vocab_size    = Config.VOCAB_SIZE
        self.merges        = {}
        self.merge_rank    = {}
        self.vocab         = {}
        self.reverse_vocab = {}
        self.special_tokens= {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
        self.next_id       = 4
        self._init_vocab()

    def _init_vocab(self):
        # Tokens byte (256)
        for i in range(256):
            tid = self.next_id
            self.next_id += 1
            self.vocab[tid]            = bytes([i])
            self.reverse_vocab[bytes([i])] = tid
        # Tokens especiais
        for token, tid in self.special_tokens.items():
            self.vocab[tid]                  = token.encode('utf-8')
            self.reverse_vocab[token.encode('utf-8')] = tid

    def train(self, texts, num_merges=3000):
        pair_counts = Counter()
        for text in texts[:1000]:
            raw    = text.encode('utf-8', errors='replace')
            tokens = [self.reverse_vocab.get(bytes([t]), 1) for t in raw]
            for i in range(len(tokens) - 1):
                pair_counts[(tokens[i], tokens[i + 1])] += 1

        ranked_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])
        for rank, (pair, _) in enumerate(ranked_pairs[:num_merges]):
            if pair in self.merges:
                continue
            new_id = self.next_id
            self.next_id += 1
            if new_id >= self.vocab_size:
                break
            self.merges[pair]      = new_id
            self.merge_rank[pair]  = rank           # salvo para uso no heap
            new_bytes = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.vocab[new_id]          = new_bytes
            self.reverse_vocab[new_bytes] = new_id

    def _apply_merges_heap(self, tokens):
        merged = list(tokens)
        n      = len(merged)
        heap   = []
        for i in range(n - 1):
            pair = (merged[i], merged[i + 1])
            if pair in self.merge_rank:
                heapq.heappush(heap, (self.merge_rank[pair], i, pair))
        while heap:
            rank, idx, pair = heapq.heappop(heap)
            if idx >= len(merged) - 1:
                continue
            if (merged[idx], merged[idx + 1]) != pair:
                continue
            merged[idx] = self.merges[pair]
            merged.pop(idx + 1)
            n -= 1
            for neighbor in [idx - 1, idx]:
                if 0 <= neighbor < n - 1:
                    new_pair = (merged[neighbor], merged[neighbor + 1])
                    if new_pair in self.merge_rank:
                        heapq.heappush(heap, (self.merge_rank[new_pair], neighbor, new_pair))
        return merged

    def encode(self, text):
        """Sem lru_cache: textos de treino variam muito, cache causaria thrash."""
        try:
            raw = text.encode('utf-8')
        except Exception:
            raw = text.encode('utf-8', errors='replace')
        tokens = [self.reverse_vocab.get(bytes([t]), 1) for t in raw]
        return tuple([self.special_tokens['<s>']] + self._apply_merges_heap(tokens))

    def decode(self, tokens):
        result = b''
        for t in tokens:
            if t in self.vocab and t >= 4:
                result += self.vocab[t]
            elif t >= 4:
                result += b'?'
        return result.decode('utf-8', errors='replace').strip()

# ====================================================================
# MODELO
# ====================================================================
class QICCRLLM:
    def __init__(self):
        self.tokenizer = BPETokenizer()
        d  = Config.D_MODEL
        df = Config.D_FF
        nl = Config.N_LAYERS

        # Estimativa do total de parâmetros
        total = (
            Config.VOCAB_SIZE * d                          # embedding
            + nl * (
                4 * d * (d + 1)                            # Q K V O
                + df * (d + 1)                             # ff_up
                + d  * (df + 1)                            # ff_down
                + 4 * d                                    # ln1 + ln2
            )
            + 2 * d                                        # ln_final
            + 1                                            # logit_scale
        )

        self.store     = WeightStore(total)
        self.grads     = array.array('f', [0.0] * total)
        self.optimizer = AdamOptimizer(total)

        allocator = WeightAllocator(total)
        self.tok_embed_off = allocator.alloc(Config.VOCAB_SIZE * d, "tok_embed")
        self.layers        = [TransformerBlock(i, allocator, self.store, True) for i in range(nl)]
        self.ln_final_off  = allocator.alloc(2 * d, "ln_final")
        self.ln_final      = LayerNorm(d, self.ln_final_off, "ln_final", True)
        self.logit_scale_off = allocator.alloc(1, "logit_scale")
        self.store.write_fp32(self.logit_scale_off, 1.0 / math.sqrt(d))

        self._init_weights()
        self.kv_cache   = KVCache(nl, Config.N_HEADS, d // Config.N_HEADS, Config.KV_MAX_SEQ)
        self.step_count = 0
        self.global_pos = 0

    # ------------------------------------------------------------------
    def _init_weights(self):
        d = Config.D_MODEL
        for i in range(Config.VOCAB_SIZE):
            for j in range(d):
                self.store.write_fp32(self.tok_embed_off + i * d + j, random.gauss(0, 0.02))
        for layer in self.layers:
            for proj in [layer.q_proj, layer.k_proj, layer.v_proj,
                         layer.o_proj, layer.ff_up, layer.ff_down]:
                self._xavier_init(proj.offset, proj.in_f, proj.out_f)
        for off in [self.ln_final_off, self.ln_final_off + d]:
            is_gamma = (off == self.ln_final_off)
            for i in range(d):
                self.store.write_fp32(off + i, 1.0 if is_gamma else 0.0)
        for layer in self.layers:
            for off in [layer.ln1.offset, layer.ln1.offset + d,
                        layer.ln2.offset, layer.ln2.offset + d]:
                is_gamma = off in [layer.ln1.offset, layer.ln2.offset]
                for i in range(d):
                    self.store.write_fp32(off + i, 1.0 if is_gamma else 0.0)

    def _xavier_init(self, offset, in_f, out_f):
        std = math.sqrt(2.0 / (in_f + out_f))
        for i in range(out_f):
            base = offset + i * (in_f + 1)
            for j in range(in_f):
                self.store.write_fp32(base + j, random.gauss(0, std))
            self.store.write_fp32(base + in_f, 0.0)

    # ------------------------------------------------------------------
    def _noam_lr(self, step):
        """
        Noam schedule conforme 'Attention Is All You Need':
            lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
        """
        d       = Config.D_MODEL
        warmup  = Config.NOAM_WARMUP
        step    = max(1, step)
        return d ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))

    # ------------------------------------------------------------------
    def _forward_impl(self, tokens, use_kv_cache=False, positions=None, training=False):
        """Forward pass.  training=False garante que dropout não é aplicado."""
        d   = Config.D_MODEL
        seq = tokens
        x   = [self.store.read_vector_fp32(self.tok_embed_off + tid * d, d) for tid in seq]

        kv          = self.kv_cache if use_kv_cache else None
        caches_list = []
        for layer in self.layers:
            x, caches = layer.forward(x, self.store,
                                       kv_cache=kv,
                                       positions=positions,
                                       training=training)
            caches_list.append(caches)

        final, ln_cache = self.ln_final.forward(x[-1], self.store)
        logit_scale = max(Config.LOGIT_SCALE_MIN,
                          min(Config.LOGIT_SCALE_MAX,
                              self.store.read_fp32(self.logit_scale_off)))
        logits = array.array('f', [0.0] * Config.VOCAB_SIZE)
        for t in range(Config.VOCAB_SIZE):
            logits[t] = logit_scale * sum(
                self.store.read_fp32(self.tok_embed_off + t * d + j) * final[j]
                for j in range(d)
            )
        return logits, caches_list, ln_cache

    # ------------------------------------------------------------------
    def prefill(self, prompt_tokens):
        self.kv_cache.clear()
        self.global_pos = 0
        positions = list(range(len(prompt_tokens)))
        logits, _, _ = self._forward_impl(
            tuple(prompt_tokens), use_kv_cache=True, positions=positions, training=False
        )
        self.global_pos = len(prompt_tokens)
        return logits

    def decode_step(self, token_id):
        logits, _, _ = self._forward_impl(
            (token_id,), use_kv_cache=True, positions=[self.global_pos], training=False
        )
        self.global_pos += 1
        return logits

    # ------------------------------------------------------------------
    def train_step(self, contexts, targets):
        """
        Batch training: contexts e targets são listas de sequências.
        Gradientes são normalizados por batch_size antes do passo do otimizador.
        """
        d          = Config.D_MODEL
        batch_size = len(contexts)
        total_loss = 0.0

        for ctx, tgt in zip(contexts, targets):
            seq     = ctx[-Config.MAX_SEQ:]
            seq_len = len(seq)
            positions = list(range(seq_len))

            x = [self.store.read_vector_fp32(self.tok_embed_off + tid * d, d) for tid in seq]
            caches_list = []
            for layer in self.layers:
                x, caches = layer.forward(x, self.store,
                                           kv_cache=None,
                                           positions=positions,
                                           training=True)
                caches_list.append(caches)

            final, ln_cache = self.ln_final.forward(x[-1], self.store)
            logit_scale = max(Config.LOGIT_SCALE_MIN,
                              min(Config.LOGIT_SCALE_MAX,
                                  self.store.read_fp32(self.logit_scale_off)))
            logits = array.array('f', [0.0] * Config.VOCAB_SIZE)
            for t in range(Config.VOCAB_SIZE):
                logits[t] = logit_scale * sum(
                    self.store.read_fp32(self.tok_embed_off + t * d + j) * final[j]
                    for j in range(d)
                )

            probs = safe_softmax(logits)

            # Label Smoothing: (1-ε) * one_hot + ε * uniform
            smooth      = Config.LABEL_SMOOTHING
            vocab       = Config.VOCAB_SIZE
            target_dist = array.array('f', [smooth / vocab] * vocab)
            target_dist[tgt] = (1.0 - smooth) + smooth / vocab

            loss = -sum(target_dist[i] * math.log(max(probs[i], 1e-9))
                        for i in range(vocab))

            # Gradiente da cross-entropy com label smoothing: p - y_smooth
            # Normalizado por batch_size para que a escala dos grads não dependa do batch
            norm        = 1.0 / batch_size
            grad_logits = array.array('f', [(probs[i] - target_dist[i]) * norm
                                             for i in range(vocab)])

            # Grad w.r.t. final (embedding de saída)
            grad_final = array.array('f', [0.0] * d)
            for t in range(vocab):
                g = grad_logits[t]
                for j in range(d):
                    w = self.store.read_fp32(self.tok_embed_off + t * d + j)
                    grad_final[j]                         += w * g * logit_scale
                    self.grads[self.tok_embed_off + t*d+j]+= final[j] * g * logit_scale

            # Grad w.r.t. logit_scale
            grad_ls = sum(
                grad_logits[t] * sum(
                    self.store.read_fp32(self.tok_embed_off + t * d + j) * final[j]
                    for j in range(d)
                )
                for t in range(vocab)
            )
            self.grads[self.logit_scale_off] += grad_ls

            # Backward LayerNorm final
            grad_ln = self.ln_final.backward(grad_final, ln_cache, self.store, self.grads)

            # Backward layers
            grad_list = [array.array('f', [0.0] * d) for _ in range(seq_len)]
            grad_list[-1] = grad_ln
            for idx in range(len(self.layers) - 1, -1, -1):
                grad_list = self.layers[idx].backward(
                    grad_list, caches_list[idx], self.store, self.grads
                )

            # Grad w.r.t. embedding de entrada (normalizado por seq_len)
            embed_norm = norm / max(1, seq_len)
            for pos, token_id in enumerate(seq):
                base = self.tok_embed_off + token_id * d
                for j in range(d):
                    self.grads[base + j] += grad_list[pos][j] * embed_norm

            total_loss += loss

        avg_loss = total_loss / batch_size

        # Lista de offsets treináveis
        trainable_offsets = [
            (self.tok_embed_off,    Config.VOCAB_SIZE * Config.D_MODEL),
            (self.ln_final_off,     2 * Config.D_MODEL),
            (self.logit_scale_off,  1),
        ]
        for layer in self.layers:
            for proj in [layer.q_proj, layer.k_proj, layer.v_proj,
                         layer.o_proj, layer.ff_up, layer.ff_down]:
                trainable_offsets.append((proj.offset, proj.W_size))
            trainable_offsets.append((layer.ln1.offset, 2 * Config.D_MODEL))
            trainable_offsets.append((layer.ln2.offset, 2 * Config.D_MODEL))

        if self.step_count % 100 == 0:
            total_gnorm = math.sqrt(
                sum(self.grads[off + i] ** 2
                    for off, size in trainable_offsets
                    for i in range(size))
            )
            current_lr = self._noam_lr(max(1, self.step_count))
            print(f"   [GradNorm] step {self.step_count}: {total_gnorm:.4f} | LR: {current_lr:.8f}")

        current_lr = self._noam_lr(max(1, self.step_count))
        self.optimizer.step(
            self.store.fp32, self.grads, trainable_offsets,
            lr=current_lr, wd=Config.WEIGHT_DECAY, clip=Config.GRAD_CLIP
        )
        self.step_count += 1
        return avg_loss

    # ------------------------------------------------------------------
    def generate_beam(self, prompt_text, max_new=80, temperature=0.7, beam_width=None):
        """
        Beam Search com KV-cache por beam.
        Cada beam mantém seu próprio cache para evitar forward quadrático.
        """
        if beam_width is None:
            beam_width = Config.BEAM_WIDTH

        EOS = self.tokenizer.special_tokens.get('</s>', 3)
        d   = Config.D_MODEL
        nl  = Config.N_LAYERS

        prompt_tokens = list(self.tokenizer.encode(prompt_text))

        # Inicializa: todos os beams partem do mesmo prefixo
        # Cada beam: (tokens, log_prob, kv_cache, global_pos, finished)
        def make_initial_beam():
            cache = KVCache(nl, Config.N_HEADS, d // Config.N_HEADS, Config.KV_MAX_SEQ)
            pos   = list(range(len(prompt_tokens)))
            # Prefill manual usando o cache deste beam
            x = [self.store.read_vector_fp32(self.tok_embed_off + tid * d, d)
                 for tid in prompt_tokens]
            for layer in self.layers:
                x, _ = layer.forward(x, self.store,
                                      kv_cache=cache, positions=pos, training=False)
            final, _ = self.ln_final.forward(x[-1], self.store)
            logit_scale = max(Config.LOGIT_SCALE_MIN,
                              min(Config.LOGIT_SCALE_MAX,
                                  self.store.read_fp32(self.logit_scale_off)))
            logits = array.array('f', [
                logit_scale * sum(
                    self.store.read_fp32(self.tok_embed_off + t * d + j) * final[j]
                    for j in range(d)
                )
                for t in range(Config.VOCAB_SIZE)
            ])
            return logits, cache, len(prompt_tokens)

        first_logits, first_cache, first_pos = make_initial_beam()

        # Obtém os top-beam_width tokens iniciais
        logits_t = array.array('f', [l / temperature for l in first_logits])
        probs    = safe_softmax(logits_t)
        topk     = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:beam_width]

        beams = []
        for tid, p in topk:
            new_cache = first_cache.clone()
            beams.append({
                'tokens':   prompt_tokens + [tid],
                'log_prob': math.log(max(p, 1e-9)),
                'cache':    new_cache,
                'pos':      first_pos,
                'finished': tid == EOS,
            })

        for _ in range(max_new - 1):
            if all(b['finished'] for b in beams):
                break

            all_candidates = []
            for b in beams:
                if b['finished']:
                    all_candidates.append(b)
                    continue

                last_token = b['tokens'][-1]
                cur_pos    = b['pos']

                # Forward de um único token com cache do beam
                x = [self.store.read_vector_fp32(self.tok_embed_off + last_token * d, d)]
                for layer in self.layers:
                    x, _ = layer.forward(x, self.store,
                                          kv_cache=b['cache'],
                                          positions=[cur_pos],
                                          training=False)
                final, _ = self.ln_final.forward(x[-1], self.store)
                logit_scale = max(Config.LOGIT_SCALE_MIN,
                                  min(Config.LOGIT_SCALE_MAX,
                                      self.store.read_fp32(self.logit_scale_off)))
                logits = array.array('f', [
                    logit_scale * sum(
                        self.store.read_fp32(self.tok_embed_off + t * d + j) * final[j]
                        for j in range(d)
                    )
                    for t in range(Config.VOCAB_SIZE)
                ])
                logits_t = array.array('f', [l / temperature for l in logits])
                probs    = safe_softmax(logits_t)
                topk     = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:beam_width]

                for tid, p in topk:
                    new_cache = b['cache'].clone()
                    all_candidates.append({
                        'tokens':   b['tokens'] + [tid],
                        'log_prob': b['log_prob'] + math.log(max(p, 1e-9)),
                        'cache':    new_cache,
                        'pos':      cur_pos + 1,
                        'finished': tid == EOS,
                    })

            # Seleciona os melhores beam_width candidatos
            all_candidates.sort(key=lambda x: x['log_prob'], reverse=True)
            beams = all_candidates[:beam_width]

        best = beams[0]
        generated = best['tokens'][len(prompt_tokens):]
        return self.tokenizer.decode(generated)

    # ------------------------------------------------------------------
    def generate(self, prompt_text, max_new=120, temperature=0.75,
                 top_k=50, top_p=0.90, use_beam=False):
        if use_beam:
            return self.generate_beam(prompt_text, max_new, temperature)

        tokens = list(self.tokenizer.encode(prompt_text))
        logits = self.prefill(tokens)

        for _ in range(max_new):
            recent = set(tokens[-32:])
            logits = array.array('f', list(logits))

            # Penalidade de repetição
            for t in recent:
                if logits[t] > 0:
                    logits[t] /= Config.REPETITION_PENALTY
                else:
                    logits[t] *= Config.REPETITION_PENALTY

            # Temperatura
            logits = array.array('f', [l / temperature for l in logits])

            # Top-k
            indexed = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)[:top_k]

            # Top-p (nucleus sampling)
            if top_p < 1.0 and indexed:
                probs  = safe_softmax(array.array('f', [t[1] for t in indexed]))
                cum    = 0.0
                cutoff = len(indexed)
                for i, p in enumerate(probs):
                    cum += p
                    if cum >= top_p:
                        cutoff = i + 1
                        break
                indexed = indexed[:max(1, cutoff)]

            probs  = safe_softmax(array.array('f', [t[1] for t in indexed]))
            r, cum = random.random(), 0.0
            chosen = indexed[-1][0]
            for i, (tid, _) in enumerate(indexed):
                cum += probs[i]
                if r <= cum:
                    chosen = tid
                    break

            tokens.append(chosen)
            if chosen == self.tokenizer.special_tokens.get('</s>', 3):
                break
            logits = self.decode_step(chosen)

        prompt_len = len(self.tokenizer.encode(prompt_text))
        return self.tokenizer.decode(tokens[prompt_len:])

    # ------------------------------------------------------------------
    def save(self, filepath="qiccr_v61"):
        with open(filepath + "_fp32.bin", 'wb') as f:
            f.write(self.store.fp32.tobytes())
        self.optimizer.save(filepath + "_optim.json.gz")
        meta = {
            'step_count': self.step_count,
            'bpe_merges': {
                f"{k[0]},{k[1]}": v
                for k, v in self.tokenizer.merges.items()
            },
            # FIX: merge_rank agora é salvo
            'bpe_merge_rank': {
                f"{k[0]},{k[1]}": v
                for k, v in self.tokenizer.merge_rank.items()
            },
            'bpe_vocab': {
                str(k): list(v)
                for k, v in self.tokenizer.vocab.items()
            },
            'next_id': self.tokenizer.next_id,
        }
        with gzip.open(filepath + "_meta.json.gz", 'wt') as f:
            json.dump(meta, f)
        print(f"💾 Salvo em {filepath}_*")

    def load(self, filepath="qiccr_v61"):
        if not os.path.exists(filepath + "_fp32.bin"):
            return False
        with open(filepath + "_fp32.bin", 'rb') as f:
            self.store.fp32 = array.array('f')
            self.store.fp32.frombytes(f.read())
        self.optimizer.load(filepath + "_optim.json.gz")
        with gzip.open(filepath + "_meta.json.gz", 'rt') as f:
            meta = json.load(f)

        self.step_count = meta['step_count']

        self.tokenizer.merges = {
            tuple(map(int, k.split(','))): v
            for k, v in meta['bpe_merges'].items()
        }
        # FIX: carrega merge_rank para que o heap funcione corretamente
        self.tokenizer.merge_rank = {
            tuple(map(int, k.split(','))): v
            for k, v in meta.get('bpe_merge_rank', {}).items()
        }
        self.tokenizer.vocab = {
            int(k): bytes(v)
            for k, v in meta['bpe_vocab'].items()
        }
        self.tokenizer.reverse_vocab = {
            v: k for k, v in self.tokenizer.vocab.items()
        }
        self.tokenizer.next_id = meta.get('next_id', 4)
        print(f"✅ Carregado de {filepath}_* (step {self.step_count})")
        return True

# ====================================================================
# TREINAMENTO
# ====================================================================
def train_model(model, filepath="treino.txt", epochs=5, max_steps=None):
    if max_steps is None:
        max_steps = Config.MAX_TRAIN_STEPS
    if not os.path.exists(filepath):
        print("❌ Arquivo não encontrado!")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    print(f"📚 {len(text):,} caracteres")

    model.tokenizer.train([text])
    tokens = list(model.tokenizer.encode(text))
    print(f"🔢 {len(tokens):,} tokens")

    best_loss = float('inf')
    for epoch in range(epochs):
        steps = min(max_steps, len(tokens) - Config.TRAIN_WINDOW_MAX - 2)
        total_loss = 0.0
        print(f"🏋️  Epoch {epoch+1}/{epochs} ({steps} passos, batch={Config.BATCH_SIZE})...")

        for i in range(steps):
            batch_contexts, batch_targets = [], []
            for _ in range(Config.BATCH_SIZE):
                window = random.randint(Config.TRAIN_WINDOW_MIN, Config.TRAIN_WINDOW_MAX)
                start  = random.randint(0, max(0, len(tokens) - window - 2))
                batch_contexts.append(tokens[start : start + window])
                batch_targets.append(tokens[start + window])

            loss = model.train_step(batch_contexts, batch_targets)
            total_loss += loss

            if (i + 1) % 500 == 0 or i == steps - 1:
                print(f"   Passo {i+1:5d}/{steps} | Loss: {total_loss/(i+1):.4f}")

        avg_loss = total_loss / steps
        print(f"✅ Epoch {epoch+1} — Loss médio: {avg_loss:.4f}")

        tag = "qiccr_v61_best" if avg_loss < best_loss - 0.001 else "qiccr_v61_latest"
        if avg_loss < best_loss - 0.001:
            best_loss = avg_loss
            print("   🏆 Novo melhor!")
        model.save(tag)

    print("🏁 Treinamento concluído!")

# ====================================================================
# CHAT INTERATIVO
# ====================================================================
def interactive_chat(model):
    print("\n🚀 QICCR-LLM v6.1")
    print("Comandos: 'sair' | 'reset' | 'beam' (toggle beam search)\n")
    use_beam = False
    while True:
        try:
            q = input("🧑 Você: ").strip()
            if not q:
                continue
            if q.lower() in ('sair', 'exit', 'quit'):
                break
            if q.lower() == 'reset':
                model.kv_cache.clear()
                model.global_pos = 0
                print("✅ Cache limpo.\n")
                continue
            if q.lower() == 'beam':
                use_beam = not use_beam
                print(f"✅ Beam search: {'ON' if use_beam else 'OFF'}\n")
                continue

            resp = model.generate(q, max_new=80, temperature=0.7, use_beam=use_beam)
            print(f"🐶 Qiccr: {resp}\n")

        except KeyboardInterrupt:
            print("\nEncerrando.")
            break

# ====================================================================
# MAIN
# ====================================================================
if __name__ == "__main__":
    random.seed(42)
    model = QICCRLLM()

    if "--train" in sys.argv:
        epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        train_model(model, epochs=epochs)
    else:
        loaded = False
        for tag in ("qiccr_v61_best", "qiccr_v61_latest", "qiccr_v6_best", "qiccr_v6_latest"):
            if os.path.exists(tag + "_fp32.bin"):
                loaded = model.load(tag)
                break
        if not loaded:
            print("⚠️  Sem checkpoint. Use --train para treinar primeiro.")
        interactive_chat(model)
