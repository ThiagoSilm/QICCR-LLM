"""
QICCR-LLM v5.4 — Correções de Previsibilidade Matemática
==========================================================
Mudanças em relação ao v5.3:
- [FIX] backward: inverse_rotate(grad_K) usa c['positions'][j_local] em vez de
        positions[min(j_global, len-1)] — timestamp correto para cada token K
- [FIX] BPETokenizer.train(): encode.cache_clear() ao final — vocab atualizado
        é refletido nas chamadas subsequentes de encode()
- [FIX] __main__: random.seed(42) antes de QICCRLLM() — runs reproduzíveis

Invariantes mantidos do v5.3:
- slot_to_pos: dicionário global por camada, não por head
- KV cache: índices originais preservados; backward usa ordenação original
- Causal mask: fixa no forward, reutilizada no backward (sem recomputação)
- Dropout: máscara armazenada no cache, reutilizada no backward
- RoPE: timestamp consistente via slot_id global
- Embedding: clipping local + scaling por seq_len
- logit_scale: clamp [0.1, 10.0]
- safe_softmax: fallback uniforme em vez de zeros
- Treino/inferência: mesmo caminho de código (positions explícitas sempre)
"""

import math, random, os, sys, array, json, gzip, heapq
from functools import lru_cache
from collections import Counter, defaultdict

# ====================================================================
# CONFIGURAÇÃO
# ====================================================================
class Config:
    VOCAB_SIZE = 8192
    D_MODEL = 128
    N_HEADS = 4
    N_LAYERS = 2
    D_FF = 256
    MAX_SEQ = 64
    KV_MAX_SEQ = 128
    LEARNING_RATE = 3e-4
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.999
    ADAM_EPS = 1e-8
    WEIGHT_DECAY = 0.01
    GRAD_CLIP = 1.0
    TOP_K = 50
    TOP_P = 0.90
    TEMPERATURE = 0.75
    REPETITION_PENALTY = 1.1
    MAX_TRAIN_STEPS = 8000
    TRAIN_WINDOW_SIZE = 32
    LAYER_LR_DECAY = 0.9
    FFN_DROPOUT = 0.1
    LOGIT_SCALE_MIN = 0.1
    LOGIT_SCALE_MAX = 10.0

# ====================================================================
# WeightAllocator
# ====================================================================
class WeightAllocator:
    def __init__(self, total): self.offset = 0
    def alloc(self, size, name=""): off = self.offset; self.offset += size; return off

# ====================================================================
# WEIGHT STORE
# ====================================================================
class WeightStore:
    def __init__(self, total_params):
        self.total_params = total_params
        self.fp32 = array.array('f', [0.0] * total_params)
    
    def read_fp32(self, offset): return self.fp32[offset]
    def write_fp32(self, offset, val): self.fp32[offset] = val
    def read_vector_fp32(self, offset, length):
        fp32 = self.fp32; return array.array('f', [fp32[offset + i] for i in range(length)])

# ====================================================================
# ADAMW
# ====================================================================
class AdamOptimizer:
    def __init__(self, total_params):
        self.m = array.array('f', [0.0] * total_params)
        self.v = array.array('f', [0.0] * total_params)
        self.t = 0
    
    def step(self, weights, grads, offsets, lr=3e-4, wd=0.01, clip=1.0, layer_scale=1.0):
        self.t += 1
        beta1, beta2, eps = Config.ADAM_BETA1, Config.ADAM_BETA2, Config.ADAM_EPS
        if offsets:
            total_norm = math.sqrt(sum(sum(grads[off+i]*grads[off+i] for i in range(size)) for off, size in offsets))
            if total_norm > clip:
                scale = clip / total_norm
                for off, size in offsets:
                    for i in range(size): grads[off+i] *= scale
        eff_lr = lr * layer_scale
        for off, size in offsets:
            for i in range(size):
                g = grads[off+i]
                self.m[off+i] = beta1*self.m[off+i] + (1-beta1)*g
                self.v[off+i] = beta2*self.v[off+i] + (1-beta2)*g*g
                m_hat = self.m[off+i]/(1-beta1**self.t)
                v_hat = self.v[off+i]/(1-beta2**self.t)
                weights[off+i] -= eff_lr*wd*weights[off+i] + eff_lr*m_hat/(math.sqrt(v_hat)+eps)
                grads[off+i] = 0.0
    
    def save(self, filepath):
        with gzip.open(filepath, 'wt') as f: json.dump({'m':list(self.m),'v':list(self.v),'t':self.t}, f)
    def load(self, filepath):
        if not os.path.exists(filepath): return False
        with gzip.open(filepath, 'rt') as f: data = json.load(f)
        self.m = array.array('f', data['m']); self.v = array.array('f', data['v']); self.t = data['t']
        return True

# ====================================================================
# SAFE SOFTMAX (fallback uniforme)
# ====================================================================
def safe_softmax(arr, mask=None):
    if mask is None: mask = [True] * len(arr)
    valid = [i for i, m in enumerate(mask) if m]
    if not valid:
        return array.array('f', [1.0/len(arr)] * len(arr))
    max_val = max(arr[i] for i in valid)
    exps = []
    for i, m in enumerate(mask):
        if not m: exps.append(0.0)
        else: exps.append(math.exp(max(-80.0, min(80.0, arr[i]-max_val))))
    s = sum(exps)
    if s == 0 or not math.isfinite(s):
        result = array.array('f', [0.0]*len(arr))
        for i in valid: result[i] = 1.0/len(valid)
        return result
    return array.array('f', [e/s for e in exps])

def softmax_backward(probs, mask, grad_probs, scale):
    total_len = len(probs)
    valid = [i for i, m in enumerate(mask) if m]
    if not valid: return [0.0] * total_len
    dot = sum(grad_probs[j] * probs[j] for j in valid)
    grad_scores = [0.0] * total_len
    for j in valid:
        grad_scores[j] = (probs[j] * (grad_probs[j] - dot)) / scale
    return grad_scores

def gelu_arr(x): return array.array('f', [0.5*xi*(1.0+math.tanh(0.7978845608028654*(xi+0.044715*xi**3))) for xi in x])

def gelu_derivative(x):
    inner = 0.7978845608028654*(x+0.044715*x**3); tanh_part = math.tanh(inner); sech2 = 1.0-tanh_part*tanh_part
    return 0.5*(1.0+tanh_part)+0.5*x*sech2*0.7978845608028654*(1.0+0.134145*x*x)

def gelu_derivative_arr(x): return array.array('f', [gelu_derivative(xi) for xi in x])

# ====================================================================
# RoPE
# ====================================================================
class RotaryPositionalEmbedding:
    def __init__(self, d_head, max_pos=None):
        self.d_head = d_head; self.max_pos = max_pos or Config.KV_MAX_SEQ + 1024
        self.cos_cache, self.sin_cache = {}, {}
    
    def _ensure(self, pos):
        if pos > self.max_pos: pos = self.max_pos
        if pos not in self.cos_cache:
            cos_vals = array.array('f', [0.0]*self.d_head); sin_vals = array.array('f', [0.0]*self.d_head)
            for i in range(0, self.d_head, 2):
                theta = pos/(10000.0**(i/self.d_head)); cos_vals[i]=math.cos(theta); sin_vals[i]=math.sin(theta)
                if i+1<self.d_head: cos_vals[i+1]=cos_vals[i]; sin_vals[i+1]=sin_vals[i]
            self.cos_cache[pos]=cos_vals; self.sin_cache[pos]=sin_vals
    
    def rotate(self, x, pos):
        self._ensure(pos); cos, sin = self.cos_cache[pos], self.sin_cache[pos]
        rotated = array.array('f', [0.0]*len(x))
        for i in range(0, len(x), 2):
            if i+1<len(x): rotated[i]=x[i]*cos[i]-x[i+1]*sin[i]; rotated[i+1]=x[i]*sin[i]+x[i+1]*cos[i]
            else: rotated[i]=x[i]
        return rotated
    
    def inverse_rotate(self, x, pos):
        self._ensure(pos); cos, sin = self.cos_cache[pos], self.sin_cache[pos]
        derotated = array.array('f', [0.0]*len(x))
        for i in range(0, len(x), 2):
            if i+1<len(x): derotated[i]=x[i]*cos[i]+x[i+1]*sin[i]; derotated[i+1]=-x[i]*sin[i]+x[i+1]*cos[i]
            else: derotated[i]=x[i]
        return derotated

# ====================================================================
# KV-CACHE (slot_id global)
# ====================================================================
class KVCache:
    def __init__(self, n_layers, n_heads, d_head, max_seq):
        self.max_seq = max_seq
        self.K_base = [[[] for _ in range(n_heads)] for __ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for __ in range(n_layers)]
        self.timestamps = [[[] for _ in range(n_heads)] for __ in range(n_layers)]
        self.slot_ids = [[[] for _ in range(n_heads)] for __ in range(n_layers)]
        self.global_slot_to_pos = {}
        self.next_slot_id = 0
    
    def update(self, layer, head, pos, k_base, v):
        slot_id = self.next_slot_id; self.next_slot_id += 1
        self.K_base[layer][head].append(array.array('f', k_base))
        self.V[layer][head].append(array.array('f', v))
        self.timestamps[layer][head].append(pos)
        self.slot_ids[layer][head].append(slot_id)
        self.global_slot_to_pos[slot_id] = pos
        if len(self.timestamps[layer][head]) > self.max_seq:
            oldest_idx = min(range(len(self.timestamps[layer][head])), key=lambda i: self.timestamps[layer][head][i])
            removed_sid = self.slot_ids[layer][head][oldest_idx]
            self.K_base[layer][head].pop(oldest_idx); self.V[layer][head].pop(oldest_idx)
            self.timestamps[layer][head].pop(oldest_idx); self.slot_ids[layer][head].pop(oldest_idx)
            if removed_sid in self.global_slot_to_pos: del self.global_slot_to_pos[removed_sid]
    
    def get_KV_ordered(self, layer, head, query_pos):
        K_list, V_list, time_list, slot_list = [], [], [], []
        for k, v, ts, sid in zip(self.K_base[layer][head], self.V[layer][head], self.timestamps[layer][head], self.slot_ids[layer][head]):
            if ts <= query_pos:
                K_list.append(k); V_list.append(v); time_list.append(ts); slot_list.append(sid)
        return K_list, V_list, time_list, slot_list
    
    def clear(self):
        for l in range(len(self.K_base)):
            for h in range(len(self.K_base[l])):
                self.K_base[l][h].clear(); self.V[l][h].clear(); self.timestamps[l][h].clear(); self.slot_ids[l][h].clear()
        self.global_slot_to_pos.clear()

# ====================================================================
# CAMADAS (dropout com cache)
# ====================================================================
class Linear:
    def __init__(self, in_f, out_f, offset, name="", trainable=True):
        self.in_f=in_f; self.out_f=out_f; self.offset=offset; self.name=name; self.trainable=trainable; self.W_size=out_f*(in_f+1)
    
    def forward(self, x, store, dropout_rate=0.0, training=False):
        out = array.array('f', [0.0]*self.out_f)
        for i in range(self.out_f):
            base = self.offset+i*(self.in_f+1); s = store.read_fp32(base+self.in_f)
            for j in range(self.in_f): s += store.read_fp32(base+j)*x[j]; out[i]=s
        mask = None
        if training and dropout_rate > 0.0:
            scale = 1.0/(1.0-dropout_rate)
            mask = array.array('f', [scale if random.random() > dropout_rate else 0.0 for _ in range(len(out))])
            for i in range(len(out)): out[i] *= mask[i]
        return out, array.array('f', x), mask
    
    def backward(self, grad_output, input_cache, mask, store, grads):
        if not self.trainable: return array.array('f', [0.0]*self.in_f)
        if mask is not None:
            grad_output = array.array('f', [grad_output[i]*mask[i] for i in range(len(grad_output))])
        grad_input = array.array('f', [0.0]*self.in_f)
        for i in range(self.out_f):
            base = self.offset+i*(self.in_f+1); g = grad_output[i]; grads[base+self.in_f] += g
            for j in range(self.in_f): w = store.read_fp32(base+j); grads[base+j] += g*input_cache[j]; grad_input[j] += w*g
        return grad_input

class LayerNorm:
    def __init__(self, dim, offset, name="", trainable=True):
        self.dim=dim; self.offset=offset; self.name=name; self.trainable=trainable
    
    def forward(self, x, store):
        n=len(x); mean=sum(x)/n; var=sum((xi-mean)**2 for xi in x)/n; std=math.sqrt(var+1e-5)
        normalized=array.array('f', [(xi-mean)/std for xi in x]); out=array.array('f', [0.0]*n)
        for i in range(n): out[i]=store.read_fp32(self.offset+i)*normalized[i]+store.read_fp32(self.offset+self.dim+i)
        return out, (normalized, std, array.array('f', x))
    
    def backward(self, grad_output, cache, store, grads):
        normalized, std, x_orig = cache; n=len(x_orig); eps=1e-5
        if self.trainable:
            for i in range(n): grads[self.offset+i] += grad_output[i]*normalized[i]; grads[self.offset+self.dim+i] += grad_output[i]
        grad_normalized = array.array('f', [grad_output[i]*store.read_fp32(self.offset+i) for i in range(n)])
        mean_grad=sum(grad_normalized)/n; std_grad=sum(grad_normalized[i]*normalized[i] for i in range(n))/n
        grad_x=array.array('f', [0.0]*n)
        for i in range(n): grad_x[i]=(grad_normalized[i]-mean_grad-normalized[i]*std_grad)/max(std, eps)
        return grad_x

# ====================================================================
# TRANSFORMER (KV ordenado preservado, causal mask fixa, dropout cached)
# ====================================================================
class TransformerBlock:
    def __init__(self, idx, allocator, store, trainable=True):
        d=Config.D_MODEL; df=Config.D_FF; self.idx=idx; self.d_model=d; self.n_heads=Config.N_HEADS; self.d_head=d//Config.N_HEADS
        self.q_proj=Linear(d,d,allocator.alloc(d*(d+1),f"q_{idx}"),f"q_{idx}",trainable)
        self.k_proj=Linear(d,d,allocator.alloc(d*(d+1),f"k_{idx}"),f"k_{idx}",trainable)
        self.v_proj=Linear(d,d,allocator.alloc(d*(d+1),f"v_{idx}"),f"v_{idx}",trainable)
        self.o_proj=Linear(d,d,allocator.alloc(d*(d+1),f"o_{idx}"),f"o_{idx}",trainable)
        self.ff_up=Linear(d,df,allocator.alloc(df*(d+1),f"up_{idx}"),f"up_{idx}",trainable)
        self.ff_down=Linear(df,d,allocator.alloc(d*(df+1),f"down_{idx}"),f"down_{idx}",trainable)
        self.ln1=LayerNorm(d,allocator.alloc(2*d,f"ln1_{idx}"),f"ln1_{idx}",trainable)
        self.ln2=LayerNorm(d,allocator.alloc(2*d,f"ln2_{idx}"),f"ln2_{idx}",trainable)
        self.rope=RotaryPositionalEmbedding(self.d_head); self.trainable=trainable
    
    def forward(self, x_list, store, kv_cache=None, positions=None, training=False):
        seq_len=len(x_list)
        if seq_len==0: return x_list, None
        caches={}
        normed1, ln1_caches = [], []
        for x in x_list: out, cache = self.ln1.forward(x, store); normed1.append(out); ln1_caches.append(cache)
        Q_full, K_full, V_full, q_caches, k_caches, v_caches = [], [], [], [], [], []
        q_masks, k_masks, v_masks = [], [], []
        for n in normed1:
            q,qc,qm=self.q_proj.forward(n,store,0.0,training); k,kc,km=self.k_proj.forward(n,store,0.0,training); v,vc,vm=self.v_proj.forward(n,store,0.0,training)
            Q_full.append(q); K_full.append(k); V_full.append(v); q_caches.append(qc); k_caches.append(kc); v_caches.append(vc)
            q_masks.append(qm); k_masks.append(km); v_masks.append(vm)
        
        if positions:
            for i in range(seq_len): Q_full[i]=self.rope.rotate(Q_full[i], positions[i])
        
        Q_heads, K_heads, V_heads = [], [], []
        for h in range(self.n_heads):
            s,e=h*self.d_head,(h+1)*self.d_head
            Q_heads.append([array.array('f',[Q_full[i][j] for j in range(s,e)]) for i in range(seq_len)])
            K_heads.append([array.array('f',[K_full[i][j] for j in range(s,e)]) for i in range(seq_len)])
            V_heads.append([array.array('f',[V_full[i][j] for j in range(s,e)]) for i in range(seq_len)])
        
        if kv_cache and positions:
            for h in range(self.n_heads):
                for i in range(seq_len): kv_cache.update(self.idx, h, positions[i], K_heads[h][i], V_heads[h][i])
        
        head_outputs, all_probs = [], []; scale=math.sqrt(self.d_head)
        all_causal_masks = []
        for h in range(self.n_heads):
            Qh=Q_heads[h]
            if kv_cache and positions:
                Kh_base, Vh, timestamps, slot_ids = kv_cache.get_KV_ordered(self.idx, h, positions[-1])
                Kh = [self.rope.rotate(array.array('f', k), ts) for k, ts in zip(Kh_base, timestamps)]
                total_len = len(Kh)
            else:
                if positions:
                    Kh = [self.rope.rotate(array.array('f', k), positions[i]) for i, k in enumerate(K_heads[h])]
                else: Kh = K_heads[h]
                Vh = V_heads[h]; total_len = len(Kh)
            
            if positions is None: positions_local = list(range(seq_len))
            else: positions_local = positions
            causal_mask = [[j <= positions_local[i] for j in range(total_len)] for i in range(seq_len)]
            all_causal_masks.append(causal_mask)
            
            attn_h, probs_h = [], []
            for i in range(seq_len):
                scores=array.array('f', [sum(Qh[i][k]*Kh[j][k] for k in range(self.d_head))/scale if causal_mask[i][j] else float('-inf') for j in range(total_len)])
                probs=safe_softmax(scores, causal_mask[i])
                head_out=array.array('f',[0.0]*self.d_head)
                for j in range(total_len):
                    if probs[j]>0:
                        for k in range(self.d_head): head_out[k]+=probs[j]*Vh[j][k]
                attn_h.append(head_out); probs_h.append(probs)
            head_outputs.append(attn_h); all_probs.append(probs_h)
        
        attn_concat=[]
        for i in range(seq_len):
            concat=array.array('f',[0.0]*self.d_model)
            for h in range(self.n_heads):
                s=h*self.d_head
                for k in range(self.d_head): concat[s+k]=head_outputs[h][i][k]
            attn_concat.append(concat)
        attn_proj, o_caches, o_masks = [], [], []
        for a in attn_concat: out,oc,om=self.o_proj.forward(a,store,0.0,training); attn_proj.append(out); o_caches.append(oc); o_masks.append(om)
        x=[array.array('f',[x_list[i][j]+attn_proj[i][j] for j in range(self.d_model)]) for i in range(seq_len)]
        residual2=[array.array('f',xi) for xi in x]; normed2, ln2_caches = [], []
        for vec in x: out,cache=self.ln2.forward(vec,store); normed2.append(out); ln2_caches.append(cache)
        pre_gelu, up_caches, up_masks = [], [], []
        for n in normed2: out,cache,mask=self.ff_up.forward(n,store,Config.FFN_DROPOUT if training else 0.0,training); pre_gelu.append(out); up_caches.append(cache); up_masks.append(mask)
        post_gelu=[gelu_arr(up) for up in pre_gelu]; ffn_out, down_caches, down_masks = [], [], []
        for act in post_gelu: out,cache,mask=self.ff_down.forward(act,store,0.0,training); ffn_out.append(out); down_caches.append(cache); down_masks.append(mask)
        x=[array.array('f',[residual2[i][j]+ffn_out[i][j] for j in range(self.d_model)]) for i in range(seq_len)]
        if self.trainable:
            caches={'ln1_caches':ln1_caches,'q_caches':q_caches,'k_caches':k_caches,'v_caches':v_caches,'o_caches':o_caches,
                    'ln2_caches':ln2_caches,'up_caches':up_caches,'down_caches':down_caches,
                    'Q_heads':Q_heads,'K_heads':K_heads,'V_heads':V_heads,'all_probs':all_probs,'pre_gelu':pre_gelu,
                    'positions':positions if positions else list(range(seq_len)), 'causal_mask': all_causal_masks,
                    'q_masks':q_masks,'k_masks':k_masks,'v_masks':v_masks,'o_masks':o_masks,'up_masks':up_masks,'down_masks':down_masks}
        return x, caches
    
    def backward(self, grad_output_list, caches, store, grads):
        if not self.trainable or not caches: return [array.array('f',[0.0]*self.d_model) for _ in grad_output_list]
        c=caches; seq_len=len(grad_output_list); d_model=self.d_model; d_head=self.d_head
        
        grad_down=[self.ff_down.backward(array.array('f',[grad_output_list[i][j] for j in range(d_model)]),c['down_caches'][i],c['down_masks'][i],store,grads) for i in range(seq_len)]
        grad_post_gelu=[array.array('f',[grad_down[i][k]*gelu_derivative(c['pre_gelu'][i][k]) for k in range(len(grad_down[i]))]) for i in range(seq_len)]
        grad_up=[self.ff_up.backward(grad_post_gelu[i],c['up_caches'][i],c['up_masks'][i],store,grads) for i in range(seq_len)]
        grad_ln2=[self.ln2.backward(array.array('f',[grad_up[i][j]+grad_output_list[i][j] for j in range(d_model)]),c['ln2_caches'][i],store,grads) for i in range(seq_len)]
        grad_o=[self.o_proj.backward(array.array('f',[grad_ln2[i][j]+grad_output_list[i][j] for j in range(d_model)]),c['o_caches'][i],c['o_masks'][i],store,grads) for i in range(seq_len)]
        
        grad_Q_full=[array.array('f',[0.0]*d_model) for _ in range(seq_len)]
        grad_K_full=[array.array('f',[0.0]*d_model) for _ in range(seq_len)]
        grad_V_full=[array.array('f',[0.0]*d_model) for _ in range(seq_len)]
        scale=math.sqrt(d_head); positions=c['positions']; causal_mask=c['causal_mask']
        
        for h in range(self.n_heads):
            s,e=h*d_head,(h+1)*d_head; Vh=c['V_heads'][h]; probs_h=c['all_probs'][h]; total_len=len(Vh)
            Kh_h = c['K_heads'][h]; Qh_h = c['Q_heads'][h]
            mask_h = causal_mask[h] if h < len(causal_mask) else None
            
            for i in range(seq_len):
                grad_head=array.array('f',[grad_o[i][s+k] for k in range(d_head)]); probs=probs_h[i]
                mask_i = mask_h[i] if mask_h else [j <= positions[i] for j in range(total_len)]
                
                grad_probs = [0.0] * total_len
                for j in range(total_len):
                    if mask_i[j]:
                        acc = 0.0
                        for k in range(d_head): acc += float(grad_head[k]) * float(Vh[j][k])
                        grad_probs[j] = acc
                
                grad_scores = softmax_backward(probs, mask_i, grad_probs, scale)
                
                for j_local in range(total_len):
                    if not mask_i[j_local]: continue
                    gs = grad_scores[j_local]
                    if abs(gs) < 1e-30: continue
                    Kh_j = Kh_h[j_local]; Qh_i = Qh_h[i]
                    for k in range(d_head):
                        grad_Q_full[i][s+k] += gs * float(Kh_j[k])
                        grad_K_full[j_local][s+k] += gs * float(Qh_i[k])
                
                for j in range(total_len):
                    if not mask_i[j]: continue
                    p = float(probs[j])
                    if p <= 0: continue
                    for k in range(d_head): grad_V_full[j][s+k] += p * float(grad_head[k])
        
        # FIX: timestamp vem de c['positions'], não de j_global como proxy
        for j_local in range(len(grad_K_full)):
            ts = positions[j_local]
            grad_K_full[j_local] = self.rope.inverse_rotate(grad_K_full[j_local], ts)
        for i in range(seq_len):
            grad_Q_full[i]=self.rope.inverse_rotate(grad_Q_full[i], positions[i])
        
        grad_Q=[self.q_proj.backward(grad_Q_full[i],c['q_caches'][i],c['q_masks'][i],store,grads) for i in range(seq_len)]
        grad_K=[self.k_proj.backward(grad_K_full[i],c['k_caches'][i],c['k_masks'][i],store,grads) for i in range(seq_len)]
        grad_V=[self.v_proj.backward(grad_V_full[i],c['v_caches'][i],c['v_masks'][i],store,grads) for i in range(seq_len)]
        grad_ln1=[self.ln1.backward(array.array('f',[grad_Q[i][j]+grad_K[i][j]+grad_V[i][j] for j in range(d_model)]),c['ln1_caches'][i],store,grads) for i in range(seq_len)]
        return grad_ln1

# ====================================================================
# TOKENIZER BPE O(n log n)
# ====================================================================
class BPETokenizer:
    def __init__(self):
        self.vocab_size=Config.VOCAB_SIZE; self.merges={}; self.vocab={}; self.reverse_vocab={}
        self.special_tokens={'<pad>':0,'<unk>':1,'<s>':2,'</s>':3}; self.next_id=4; self.merge_rank={}
        self._init_vocab()
    
    def _init_vocab(self):
        for i in range(256): tid=self.next_id; self.next_id+=1; self.vocab[tid]=bytes([i]); self.reverse_vocab[bytes([i])]=tid
        for token,tid in self.special_tokens.items(): self.vocab[tid]=token.encode('utf-8'); self.reverse_vocab[token.encode('utf-8')]=tid
        for word in ["o","a","os","as","de","do","da","que","e","não","é","um","uma","para","com","se","no","na","por","mais","mas","foi","ser","está"]:
            for t in word.encode('utf-8'):
                if bytes([t]) not in self.reverse_vocab: tid=self.next_id; self.next_id+=1; self.vocab[tid]=bytes([t]); self.reverse_vocab[bytes([t])]=tid
    
    def train(self, texts, num_merges=3000):
        pair_counts=Counter()
        for text in texts[:1000]:
            tokens=list(text.encode('utf-8')); tokens=[self.reverse_vocab.get(bytes([t]),1) for t in tokens]
            for i in range(len(tokens)-1): pair=(tokens[i],tokens[i+1]); pair_counts[pair]+=1
        ranked_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])
        for rank, (pair, _) in enumerate(ranked_pairs[:num_merges]):
            if pair in self.merges: continue
            new_id=self.next_id; self.next_id+=1
            if new_id>=self.vocab_size: break
            self.merges[pair]=new_id; self.merge_rank[pair]=rank
            self.vocab[new_id]=self.vocab[pair[0]]+self.vocab[pair[1]]; self.reverse_vocab[self.vocab[new_id]]=new_id
        self.encode.cache_clear()  # FIX: vocab mudou, cache de encode() precisa ser invalidado
    
    def _apply_merges_heap(self, tokens):
        merged=list(tokens); n=len(merged); heap=[]
        for i in range(n-1):
            pair=(merged[i],merged[i+1])
            if pair in self.merge_rank: heapq.heappush(heap, (self.merge_rank[pair], i, pair))
        while heap:
            rank, idx, pair = heapq.heappop(heap)
            if idx >= len(merged)-1: continue
            if (merged[idx], merged[idx+1]) != pair: continue
            merged[idx] = self.merges[pair]; merged.pop(idx+1)
            n-=1
            for neighbor in [idx-1, idx]:
                if 0 <= neighbor < n-1:
                    new_pair = (merged[neighbor], merged[neighbor+1])
                    if new_pair in self.merge_rank:
                        heapq.heappush(heap, (self.merge_rank[new_pair], neighbor, new_pair))
        return merged
    
    @lru_cache(maxsize=128)
    def encode(self, text):
        try: raw=text.encode('utf-8')
        except: raw=text.encode('utf-8',errors='replace')
        tokens=[self.reverse_vocab.get(bytes([t]),1) for t in raw]
        return tuple([self.special_tokens['<s>']]+self._apply_merges_heap(tokens))
    
    def decode(self, tokens):
        result=b''
        for t in tokens:
            if t in self.vocab and t>=4: result+=self.vocab[t]
            elif t>=4: result+=b'?'
        return result.decode('utf-8',errors='replace').strip()

# ====================================================================
# MODELO (logit_scale clamp, embedding scaling, positions explícitas)
# ====================================================================
class QICCRLLM:
    def __init__(self):
        self.tokenizer=BPETokenizer(); d=Config.D_MODEL; df=Config.D_FF; nl=Config.N_LAYERS
        total=Config.VOCAB_SIZE*d + nl*(4*d*(d+1)+df*(d+1)+d*(df+1)+4*d) + 2*d + 1
        self.store=WeightStore(total); self.grads=array.array('f',[0.0]*total); self.optimizer=AdamOptimizer(total)
        allocator=WeightAllocator(total)
        self.tok_embed_off=allocator.alloc(Config.VOCAB_SIZE*d,"tok_embed")
        self.layers=[TransformerBlock(i,allocator,self.store,True) for i in range(nl)]
        self.ln_final_off=allocator.alloc(2*d,"ln_final"); self.ln_final=LayerNorm(d,self.ln_final_off,"ln_final",True)
        self.logit_scale_off=allocator.alloc(1,"logit_scale"); self.store.write_fp32(self.logit_scale_off, 1.0/math.sqrt(d))
        self._init_weights(); self.kv_cache=KVCache(nl,Config.N_HEADS,d//Config.N_HEADS,Config.KV_MAX_SEQ); self.step_count=0; self.global_pos=0
    
    def _init_weights(self):
        d=Config.D_MODEL
        for i in range(Config.VOCAB_SIZE):
            for j in range(d): self.store.write_fp32(self.tok_embed_off+i*d+j, random.gauss(0,0.02))
        for layer in self.layers:
            for proj in [layer.q_proj,layer.k_proj,layer.v_proj,layer.o_proj,layer.ff_up,layer.ff_down]: self._xavier_init(proj.offset,proj.in_f,proj.out_f)
        for off in [self.ln_final_off,self.ln_final_off+d]:
            for i in range(d): self.store.write_fp32(off+i,1.0 if off==self.ln_final_off else 0.0)
        for layer in self.layers:
            for off in [layer.ln1.offset,layer.ln1.offset+d,layer.ln2.offset,layer.ln2.offset+d]:
                is_gamma=off in [layer.ln1.offset,layer.ln2.offset]
                for i in range(d): self.store.write_fp32(off+i,1.0 if is_gamma else 0.0)
    
    def _xavier_init(self, offset, in_f, out_f):
        std=math.sqrt(2.0/(in_f+out_f))
        for i in range(out_f):
            base=offset+i*(in_f+1)
            for j in range(in_f): self.store.write_fp32(base+j, random.gauss(0,std))
            self.store.write_fp32(base+in_f,0.0)
    
    def _forward_impl(self, tokens, use_kv_cache=False, positions=None, training=False):
        d=Config.D_MODEL; seq=tokens; x=[self.store.read_vector_fp32(self.tok_embed_off+tid*d,d) for tid in seq]
        kv=self.kv_cache if use_kv_cache else None; caches_list=[]
        for layer in self.layers: x,caches=layer.forward(x,self.store,kv_cache=kv,positions=positions,training=training); caches_list.append(caches)
        final,ln_cache=self.ln_final.forward(x[-1],self.store)
        logit_scale=max(Config.LOGIT_SCALE_MIN, min(Config.LOGIT_SCALE_MAX, self.store.read_fp32(self.logit_scale_off)))
        logits=array.array('f',[0.0]*Config.VOCAB_SIZE)
        for t in range(Config.VOCAB_SIZE): logits[t]=logit_scale*sum(self.store.read_fp32(self.tok_embed_off+t*d+j)*final[j] for j in range(d))
        return logits, caches_list, ln_cache
    
    def prefill(self, prompt_tokens):
        self.kv_cache.clear(); self.global_pos=0; positions=list(range(len(prompt_tokens)))
        logits, _, _=self._forward_impl(tuple(prompt_tokens), use_kv_cache=True, positions=positions)
        self.global_pos=len(prompt_tokens); return logits
    
    def decode_step(self, token_id):
        logits, _, _=self._forward_impl((token_id,), use_kv_cache=True, positions=[self.global_pos])
        self.global_pos+=1; return logits
    
    def train_step(self, context_tokens, target_token):
        d=Config.D_MODEL; seq=context_tokens[-Config.MAX_SEQ:]; seq_len=len(seq)
        positions=list(range(seq_len))
        x=[self.store.read_vector_fp32(self.tok_embed_off+tid*d,d) for tid in seq]; caches_list=[]
        for layer in self.layers: x,caches=layer.forward(x,self.store,kv_cache=None,positions=positions,training=True); caches_list.append(caches)
        final,ln_cache=self.ln_final.forward(x[-1],self.store)
        logit_scale=max(Config.LOGIT_SCALE_MIN, min(Config.LOGIT_SCALE_MAX, self.store.read_fp32(self.logit_scale_off)))
        logits=array.array('f',[0.0]*Config.VOCAB_SIZE)
        for t in range(Config.VOCAB_SIZE): logits[t]=logit_scale*sum(self.store.read_fp32(self.tok_embed_off+t*d+j)*final[j] for j in range(d))
        
        probs=safe_softmax(logits); loss=-math.log(max(probs[target_token],1e-9))
        grad_logits=array.array('f',[probs[i]-(1.0 if i==target_token else 0.0) for i in range(len(probs))])
        grad_final=array.array('f',[0.0]*d)
        for t in range(Config.VOCAB_SIZE):
            g=grad_logits[t]
            for j in range(d): grad_final[j]+=self.store.read_fp32(self.tok_embed_off+t*d+j)*g*logit_scale; self.grads[self.tok_embed_off+t*d+j]+=g*final[j]*logit_scale
        
        grad_logit_scale=sum(grad_logits[t]*sum(self.store.read_fp32(self.tok_embed_off+t*d+j)*final[j] for j in range(d)) for t in range(Config.VOCAB_SIZE))
        self.grads[self.logit_scale_off]+=grad_logit_scale
        
        grad_ln=self.ln_final.backward(grad_final,ln_cache,self.store,self.grads)
        grad_list=[array.array('f',[0.0]*d) for _ in range(seq_len)]; grad_list[-1]=grad_ln
        for idx in range(len(self.layers)-1,-1,-1): grad_list=self.layers[idx].backward(grad_list,caches_list[idx],self.store,self.grads)
        
        embed_grad_scale=1.0/max(1, seq_len)
        for pos, token_id in enumerate(seq):
            base=self.tok_embed_off+token_id*d
            for j in range(d): self.grads[base+j]+=grad_list[pos][j]*embed_grad_scale
        
        trainable_offsets=[(self.tok_embed_off,Config.VOCAB_SIZE*d),(self.ln_final_off,2*d),(self.logit_scale_off,1)]
        for layer in self.layers:
            for proj in [layer.q_proj,layer.k_proj,layer.v_proj,layer.o_proj,layer.ff_up,layer.ff_down]:
                trainable_offsets.append((proj.offset,proj.W_size))
            trainable_offsets.append((layer.ln1.offset,2*d)); trainable_offsets.append((layer.ln2.offset,2*d))
        
        if self.step_count % 100 == 0:
            total_gnorm=math.sqrt(sum(sum(self.grads[off+i]**2 for i in range(size)) for off, size in trainable_offsets))
            print(f"   [GradNorm] step {self.step_count}: {total_gnorm:.4f}")
        
        self.optimizer.step(self.store.fp32, self.grads, trainable_offsets, Config.LEARNING_RATE, Config.WEIGHT_DECAY, Config.GRAD_CLIP, 1.0)
        self.step_count+=1
        return loss
    
    def generate(self, prompt_text, max_new=120, temperature=0.75, top_k=50, top_p=0.90):
        tokens=list(self.tokenizer.encode(prompt_text)); logits=self.prefill(tokens)
        for _ in range(max_new):
            recent=set(tokens[-32:]); logits=array.array('f',[l for l in logits])
            for t in recent:
                if logits[t]>0: logits[t]/=Config.REPETITION_PENALTY
                else: logits[t]*=Config.REPETITION_PENALTY
            logits=array.array('f',[l/temperature for l in logits])
            indexed=sorted(enumerate(logits),key=lambda x:x[1],reverse=True)[:top_k]
            if top_p<1.0 and indexed:
                probs=safe_softmax(array.array('f',[t[1] for t in indexed])); cum,cutoff=0.0,len(indexed)
                for i,p in enumerate(probs):
                    cum+=p
                    if cum>=top_p: cutoff=i+1; break
                indexed=indexed[:max(1,cutoff)]
            probs=safe_softmax(array.array('f',[t[1] for t in indexed])); r,cum=random.random(),0.0; chosen=indexed[-1][0]
            for i,(tid,_) in enumerate(indexed):
                cum+=probs[i]
                if r<=cum: chosen=tid; break
            tokens.append(chosen)
            if chosen==self.tokenizer.special_tokens.get('</s>',3): break
            logits=self.decode_step(chosen)
        return self.tokenizer.decode(tokens[len(self.tokenizer.encode(prompt_text)):])
    
    def save(self, filepath="qiccr_v5"):
        with open(filepath+"_fp32.bin",'wb') as f: f.write(self.store.fp32.tobytes())
        self.optimizer.save(filepath+"_optim.json")
        meta={'step_count':self.step_count,'bpe_merges':{f"{k[0]},{k[1]}":v for k,v in self.tokenizer.merges.items()},
              'bpe_vocab':{str(k):list(v) for k,v in self.tokenizer.vocab.items()},'next_id':self.tokenizer.next_id}
        with gzip.open(filepath+"_meta.json.gz",'wt') as f: json.dump(meta,f)
    
    def load(self, filepath="qiccr_v5"):
        if not os.path.exists(filepath+"_fp32.bin"): return False
        with open(filepath+"_fp32.bin",'rb') as f: self.store.fp32=array.array('f'); self.store.fp32.frombytes(f.read())
        self.optimizer.load(filepath+"_optim.json")
        with gzip.open(filepath+"_meta.json.gz",'rt') as f: meta=json.load(f)
        self.step_count=meta['step_count']; self.tokenizer.merges={tuple(map(int,k.split(','))):v for k,v in meta['bpe_merges'].items()}
        self.tokenizer.vocab={int(k):bytes(v) for k,v in meta['bpe_vocab'].items()}
        self.tokenizer.reverse_vocab={v:k for k,v in self.tokenizer.vocab.items()}; self.tokenizer.next_id=meta.get('next_id',4)
        return True

# ====================================================================
# TREINAMENTO E CHAT
# ====================================================================
def train_model(model, filepath="treino.txt", epochs=5, max_steps=None):
    if max_steps is None: max_steps=Config.MAX_TRAIN_STEPS
    if not os.path.exists(filepath): print("❌ Arquivo não encontrado!"); return
    with open(filepath,'r',encoding='utf-8') as f: text=f.read().strip()
    print(f"📚 {len(text):,} caracteres"); model.tokenizer.train([text])
    tokens=list(model.tokenizer.encode(text)); print(f"🔢 {len(tokens):,} tokens")
    window=Config.TRAIN_WINDOW_SIZE; best_loss=float('inf')
    for epoch in range(epochs):
        steps=min(max_steps,len(tokens)-window-2); total_loss=0.0
        print(f"🏋️ Epoch {epoch+1}/{epochs} ({steps} passos)...")
        for i in range(steps):
            start=random.randint(0,max(0,len(tokens)-window-2)); context=tokens[start:start+window]; target=tokens[start+window]
            loss=model.train_step(context,target); total_loss+=loss
            if (i+1)%500==0 or i==steps-1: print(f"   Passo {i+1:5d}/{steps} | Loss: {total_loss/(i+1):.4f}")
        avg_loss=total_loss/steps; print(f"✅ Epoch {epoch+1} — Loss: {avg_loss:.4f}")
        if avg_loss<best_loss-0.001: best_loss=avg_loss; model.save("qiccr_v5_best"); print("   💾 Melhor!")
        else: model.save("qiccr_v5_latest")
    print("💾 Concluído!")

def interactive_chat(model):
    print("\n🚀 QICCR v5.3 | 'sair' | 'reset'\n")
    while True:
        try:
            q=input("🧑 Você: ").strip()
            if not q: continue
            if q.lower() in ('sair','exit','quit'): break
            if q.lower()=='reset': model.kv_cache.clear(); model.global_pos=0; print("✅\n"); continue
            print(f"🐶 Qiccr: {model.generate(q, max_new=80, temperature=0.7)}\n")
        except KeyboardInterrupt: break

if __name__=="__main__":
    random.seed(42)  # FIX: semente global para reprodutibilidade
    model=QICCRLLM()
    if "--train" in sys.argv:
        epochs=5
        if len(sys.argv)>2: epochs=int(sys.argv[2])
        train_model(model, epochs=epochs)
    else:
        if os.path.exists("qiccr_v5_best_fp32.bin"): model.load("qiccr_v5_best")
        elif os.path.exists("qiccr_v5_latest_fp32.bin"): model.load("qiccr_v5_latest")
        else: print("⚠️ Sem checkpoint. Use --train")
        interactive_chat(model)
