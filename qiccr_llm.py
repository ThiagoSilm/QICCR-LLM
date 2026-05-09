"""
QICCR-LLM
"""

import math, random, os, sys, array, json, gzip, heapq
from collections import Counter

# ====================================================================
# CONFIGURAÇÃO
# ====================================================================
class Config:
    VOCAB_SIZE       = 8192
    MAX_SEQ          = 64
    KV_MAX_SEQ       = 128
    ADAM_BETA1       = 0.9
    ADAM_BETA2       = 0.999
    ADAM_EPS         = 1e-8
    WEIGHT_DECAY     = 0.01
    GRAD_CLIP        = 1.0
    TOP_K            = 50
    TOP_P            = 0.90
    TEMPERATURE      = 0.75
    REPETITION_PENALTY = 1.1
    BATCH_SIZE       = 4
    FFN_DROPOUT      = 0.1
    LOGIT_SCALE_MIN  = 0.1
    LOGIT_SCALE_MAX  = 10.0
    LABEL_SMOOTHING  = 0.1
    NOAM_WARMUP      = 4000
    BEAM_WIDTH       = 3
    
    # Estágio 1
    S1_D_MODEL  = 64
    S1_N_HEADS  = 2
    S1_N_LAYERS = 1
    S1_D_FF     = 128
    S1_STEPS    = 2000
    
    # Estágio 2 (final)
    S2_D_MODEL  = 128
    S2_N_HEADS  = 4
    S2_N_LAYERS = 2
    S2_D_FF     = 256
    S2_STEPS    = 4000
    
    # Estágio 3 (fine-tuning)
    S3_STEPS    = 2000
    
    TRAIN_WINDOW_MIN = 16
    TRAIN_WINDOW_MAX = 48

# ====================================================================
# WeightAllocator
# ====================================================================
class WeightAllocator:
    def __init__(self, total):
        self.total = total
        self.offset = 0
    def alloc(self, size, name=""):
        off = self.offset
        self.offset += size
        assert self.offset <= self.total, f"Overflow '{name}': {self.offset} > {self.total}"
        return off

# ====================================================================
# WEIGHT STORE
# ====================================================================
class WeightStore:
    def __init__(self, total_params):
        self.total_params = total_params
        self.fp32 = array.array('f', [0.0] * total_params)
    def read_fp32(self, off): return self.fp32[off]
    def write_fp32(self, off, val): self.fp32[off] = val
    def read_vector_fp32(self, off, length):
        return array.array('f', [self.fp32[off+i] for i in range(length)])

# ====================================================================
# ADAMW
# ====================================================================
class AdamOptimizer:
    def __init__(self, total_params):
        self.m = array.array('f', [0.0]*total_params)
        self.v = array.array('f', [0.0]*total_params)
        self.t = 0
    
    def step(self, weights, grads, offsets, lr, wd, clip):
        self.t += 1
        b1, b2, eps = Config.ADAM_BETA1, Config.ADAM_BETA2, Config.ADAM_EPS
        if offsets:
            total_norm = math.sqrt(sum(grads[off+i]**2 for off,sz in offsets for i in range(sz)))
            if total_norm > clip:
                sc = clip/total_norm
                for off,sz in offsets:
                    for i in range(sz): grads[off+i] *= sc
        for off,sz in offsets:
            for i in range(sz):
                g = grads[off+i]
                self.m[off+i] = b1*self.m[off+i] + (1-b1)*g
                self.v[off+i] = b2*self.v[off+i] + (1-b2)*g*g
                mh = self.m[off+i]/(1-b1**self.t)
                vh = self.v[off+i]/(1-b2**self.t)
                weights[off+i] -= lr*wd*weights[off+i] + lr*mh/(math.sqrt(vh)+eps)
                grads[off+i] = 0.0
    
    def reset_moments(self, offsets):
        for off,sz in offsets:
            for i in range(sz):
                self.m[off+i] = 0.0
                self.v[off+i] = 0.0
    
    def save(self, path):
        with gzip.open(path,'wt') as f: json.dump({'m':list(self.m),'v':list(self.v),'t':self.t}, f)
    def load(self, path):
        if not os.path.exists(path): return False
        with gzip.open(path,'rt') as f: d=json.load(f)
        self.m=array.array('f',d['m']); self.v=array.array('f',d['v']); self.t=d['t']
        return True

# ====================================================================
# FUNÇÕES AUXILIARES
# ====================================================================
def safe_softmax(arr, mask=None):
    if mask is None: mask=[True]*len(arr)
    valid=[i for i,m in enumerate(mask) if m]
    if not valid: return array.array('f',[1.0/len(arr)]*len(arr))
    mx=max(arr[i] for i in valid)
    exps=[math.exp(max(-80,min(80,arr[i]-mx))) if m else 0.0 for i,m in enumerate(mask)]
    s=sum(exps)
    if s==0 or not math.isfinite(s):
        res=array.array('f',[0.0]*len(arr))
        for i in valid: res[i]=1.0/len(valid)
        return res
    return array.array('f',[e/s for e in exps])

def softmax_backward(probs, mask, grad_probs, scale):
    valid=[i for i,m in enumerate(mask) if m]
    if not valid: return [0.0]*len(probs)
    dot=sum(grad_probs[j]*probs[j] for j in valid)
    return [(probs[j]*(grad_probs[j]-dot))/scale if m else 0.0 for j,m in enumerate(mask)]

def gelu_arr(x): return array.array('f',[0.5*xi*(1.0+math.tanh(0.7978845608028654*(xi+0.044715*xi**3))) for xi in x])

def gelu_derivative_arr(x):
    def d(xi):
        inner=0.7978845608028654*(xi+0.044715*xi**3); th=math.tanh(inner); s2=1.0-th*th
        return 0.5*(1.0+th)+0.5*xi*s2*0.7978845608028654*(1.0+0.134145*xi*xi)
    return array.array('f',[d(xi) for xi in x])

# ====================================================================
# RoPE
# ====================================================================
class RotaryPositionalEmbedding:
    def __init__(self, d_head, max_pos=None):
        self.d_head=d_head; self.max_pos=max_pos or Config.KV_MAX_SEQ+1024
        self.cos_cache,self.sin_cache={},{}
    def _ensure(self, pos):
        pos=min(pos,self.max_pos)
        if pos not in self.cos_cache:
            cv=array.array('f',[0.0]*self.d_head); sv=array.array('f',[0.0]*self.d_head)
            for i in range(0,self.d_head,2):
                th=pos/(10000.0**(i/self.d_head)); c,s=math.cos(th),math.sin(th)
                cv[i]=c; sv[i]=s
                if i+1<self.d_head: cv[i+1]=c; sv[i+1]=s
            self.cos_cache[pos]=cv; self.sin_cache[pos]=sv
    def rotate(self, x, pos):
        self._ensure(pos); c,s=self.cos_cache[pos],self.sin_cache[pos]
        r=array.array('f',[0.0]*len(x))
        for i in range(0,len(x),2):
            if i+1<len(x): r[i]=x[i]*c[i]-x[i+1]*s[i]; r[i+1]=x[i]*s[i]+x[i+1]*c[i]
            else: r[i]=x[i]
        return r
    def inverse_rotate(self, x, pos):
        self._ensure(pos); c,s=self.cos_cache[pos],self.sin_cache[pos]
        r=array.array('f',[0.0]*len(x))
        for i in range(0,len(x),2):
            if i+1<len(x): r[i]=x[i]*c[i]+x[i+1]*s[i]; r[i+1]=-x[i]*s[i]+x[i+1]*c[i]
            else: r[i]=x[i]
        return r

# ====================================================================
# KV-CACHE
# ====================================================================
class KVCache:
    def __init__(self, n_layers, n_heads, d_head, max_seq):
        self.n_layers=n_layers; self.n_heads=n_heads; self.d_head=d_head; self.max_seq=max_seq
        self._reset()
    def _reset(self):
        self.K_base=[[[] for _ in range(self.n_heads)] for __ in range(self.n_layers)]
        self.V=[[[] for _ in range(self.n_heads)] for __ in range(self.n_layers)]
        self.timestamps=[[[] for _ in range(self.n_heads)] for __ in range(self.n_layers)]
    def update(self, layer, head, pos, k_base, v):
        self.K_base[layer][head].append(array.array('f',k_base))
        self.V[layer][head].append(array.array('f',v))
        self.timestamps[layer][head].append(pos)
        if len(self.timestamps[layer][head])>self.max_seq:
            oldest=min(range(len(self.timestamps[layer][head])), key=lambda i: self.timestamps[layer][head][i])
            self.K_base[layer][head].pop(oldest); self.V[layer][head].pop(oldest); self.timestamps[layer][head].pop(oldest)
    def get_KV_ordered(self, layer, head, query_pos):
        K,V,T=[],[],[]
        for k,v,ts in zip(self.K_base[layer][head],self.V[layer][head],self.timestamps[layer][head]):
            if ts<=query_pos: K.append(k); V.append(v); T.append(ts)
        return K,V,T
    def clone(self):
        new=KVCache(self.n_layers,self.n_heads,self.d_head,self.max_seq)
        for l in range(self.n_layers):
            for h in range(self.n_heads):
                new.K_base[l][h]=[array.array('f',k) for k in self.K_base[l][h]]
                new.V[l][h]=[array.array('f',v) for v in self.V[l][h]]
                new.timestamps[l][h]=list(self.timestamps[l][h])
        return new
    def clear(self): self._reset()

# ====================================================================
# CAMADAS
# ====================================================================
class Linear:
    def __init__(self, in_f, out_f, offset, name="", trainable=True):
        self.in_f=in_f; self.out_f=out_f; self.offset=offset; self.name=name
        self.trainable=trainable; self.W_size=out_f*(in_f+1)
    def forward(self, x, store, dropout_rate=0.0, training=False):
        out=array.array('f',[0.0]*self.out_f)
        for i in range(self.out_f):
            base=self.offset+i*(self.in_f+1); s=store.read_fp32(base+self.in_f)
            for j in range(self.in_f): s+=store.read_fp32(base+j)*x[j]
            out[i]=s
        mask=None
        if training and dropout_rate>0.0:
            scale=1.0/(1.0-dropout_rate)
            mask=array.array('f',[scale if random.random()>dropout_rate else 0.0 for _ in range(len(out))])
            for i in range(len(out)): out[i]*=mask[i]
        return out, array.array('f',x), mask
    def backward(self, grad_output, input_cache, mask, store, grads):
        if not self.trainable: return array.array('f',[0.0]*self.in_f)
        if mask is not None: grad_output=array.array('f',[grad_output[i]*mask[i] for i in range(len(grad_output))])
        gi=array.array('f',[0.0]*self.in_f)
        for i in range(self.out_f):
            base=self.offset+i*(self.in_f+1); g=grad_output[i]; grads[base+self.in_f]+=g
            for j in range(self.in_f): w=store.read_fp32(base+j); grads[base+j]+=g*input_cache[j]; gi[j]+=w*g
        return gi

class LayerNorm:
    def __init__(self, dim, offset, name="", trainable=True):
        self.dim=dim; self.offset=offset; self.name=name; self.trainable=trainable
    def forward(self, x, store):
        n=len(x); mean=sum(x)/n; var=sum((xi-mean)**2 for xi in x)/n; std=math.sqrt(var+1e-5)
        norm=array.array('f',[(xi-mean)/std for xi in x])
        out=array.array('f',[store.read_fp32(self.offset+i)*norm[i]+store.read_fp32(self.offset+self.dim+i) for i in range(n)])
        return out, (norm, std, array.array('f',x))
    def backward(self, go, cache, store, grads):
        norm,std,x_orig=cache; n=len(x_orig); eps=1e-5
        if self.trainable:
            for i in range(n): grads[self.offset+i]+=go[i]*norm[i]; grads[self.offset+self.dim+i]+=go[i]
        gn=array.array('f',[go[i]*store.read_fp32(self.offset+i) for i in range(n)])
        mg=sum(gn)/n; sg=sum(gn[i]*norm[i] for i in range(n))/n
        return array.array('f',[(gn[i]-mg-norm[i]*sg)/max(std,eps) for i in range(n)])

# ====================================================================
# TRANSFORMER BLOCK
# ====================================================================
class TransformerBlock:
    def __init__(self, idx, alloc, store, d_model, n_heads, d_ff, trainable=True):
        self.idx=idx; self.d_model=d_model; self.n_heads=n_heads; self.d_head=d_model//n_heads
        self.trainable=trainable
        self.q_proj=Linear(d_model,d_model,alloc.alloc(d_model*(d_model+1),f"q_{idx}"),f"q_{idx}",trainable)
        self.k_proj=Linear(d_model,d_model,alloc.alloc(d_model*(d_model+1),f"k_{idx}"),f"k_{idx}",trainable)
        self.v_proj=Linear(d_model,d_model,alloc.alloc(d_model*(d_model+1),f"v_{idx}"),f"v_{idx}",trainable)
        self.o_proj=Linear(d_model,d_model,alloc.alloc(d_model*(d_model+1),f"o_{idx}"),f"o_{idx}",trainable)
        self.ff_up=Linear(d_model,d_ff,alloc.alloc(d_ff*(d_model+1),f"up_{idx}"),f"up_{idx}",trainable)
        self.ff_down=Linear(d_ff,d_model,alloc.alloc(d_model*(d_ff+1),f"dn_{idx}"),f"dn_{idx}",trainable)
        self.ln1=LayerNorm(d_model,alloc.alloc(2*d_model,f"ln1_{idx}"),f"ln1_{idx}",trainable)
        self.ln2=LayerNorm(d_model,alloc.alloc(2*d_model,f"ln2_{idx}"),f"ln2_{idx}",trainable)
        self.rope=RotaryPositionalEmbedding(self.d_head)
        self.sz_q=d_model*(d_model+1); self.sz_ln=2*d_model
        self.sz_up=d_ff*(d_model+1); self.sz_down=d_model*(d_ff+1)
    
    def forward(self, x_list, store, kv_cache=None, positions=None, training=False):
        seq_len=len(x_list)
        if seq_len==0: return x_list, None
        d_model,n_heads,d_head=self.d_model,self.n_heads,self.d_head
        scale=math.sqrt(d_head)
        normed1,ln1_caches=[],[]
        for x in x_list: o,c=self.ln1.forward(x,store); normed1.append(o); ln1_caches.append(c)
        Q,K,V,qc,kc,vc,qm,km,vm=[],[],[],[],[],[],[],[],[]
        for n in normed1:
            q,cq,mq=self.q_proj.forward(n,store,0.0,training); k,ck,mk=self.k_proj.forward(n,store,0.0,training); v,cv,mv=self.v_proj.forward(n,store,0.0,training)
            Q.append(q); K.append(k); V.append(v); qc.append(cq); kc.append(ck); vc.append(cv); qm.append(mq); km.append(mk); vm.append(mv)
        pos_local=positions if positions is not None else list(range(seq_len))
        for i in range(seq_len): Q[i]=self.rope.rotate(Q[i],pos_local[i])
        Qh,Kh,Vh=[],[],[]
        for h in range(n_heads):
            s,e=h*d_head,(h+1)*d_head
            Qh.append([array.array('f',[Q[i][j] for j in range(s,e)]) for i in range(seq_len)])
            Kh.append([array.array('f',[K[i][j] for j in range(s,e)]) for i in range(seq_len)])
            Vh.append([array.array('f',[V[i][j] for j in range(s,e)]) for i in range(seq_len)])
        if kv_cache and positions:
            for h in range(n_heads):
                for i in range(seq_len): kv_cache.update(self.idx,h,pos_local[i],Kh[h][i],Vh[h][i])
        ho,ap,am=[],[],[]
        for h in range(n_heads):
            Qh_h=Qh[h]
            if kv_cache and positions:
                Kb,Vb,Tb=kv_cache.get_KV_ordered(self.idx,h,pos_local[-1])
                Kh_h=[self.rope.rotate(array.array('f',k),ts) for k,ts in zip(Kb,Tb)]
                Vh_h=Vb; total_len=len(Kh_h)
                mh=[[Tb[j]<=pos_local[i] for j in range(total_len)] for i in range(seq_len)]
            else:
                Kh_h=[self.rope.rotate(array.array('f',k),pos_local[i]) for i,k in enumerate(Kh[h])]
                Vh_h=Vh[h]; total_len=len(Kh_h)
                mh=[[j<=i for j in range(total_len)] for i in range(seq_len)]
            am.append(mh)
            ah,ph=[],[]
            for i in range(seq_len):
                scores=array.array('f',[sum(Qh_h[i][k]*Kh_h[j][k] for k in range(d_head))/scale if mh[i][j] else float('-inf') for j in range(total_len)])
                probs=safe_softmax(scores,mh[i])
                head_out=array.array('f',[0.0]*d_head)
                for j in range(total_len):
                    if probs[j]>0:
                        for k in range(d_head): head_out[k]+=probs[j]*Vh_h[j][k]
                ah.append(head_out); ph.append(probs)
            ho.append(ah); ap.append(ph)
        ac=[]
        for i in range(seq_len):
            concat=array.array('f',[0.0]*d_model)
            for h in range(n_heads):
                s=h*d_head
                for k in range(d_head): concat[s+k]=ho[h][i][k]
            ac.append(concat)
        ao,oc,om=[],[],[]
        for a in ac: o,c,m=self.o_proj.forward(a,store,0.0,training); ao.append(o); oc.append(c); om.append(m)
        x=[array.array('f',[x_list[i][j]+ao[i][j] for j in range(d_model)]) for i in range(seq_len)]
        r2=[array.array('f',xi) for xi in x]
        n2,c2=[],[]
        for vec in x: o,c=self.ln2.forward(vec,store); n2.append(o); c2.append(c)
        pre,uc,um2=[],[],[]
        for n in n2: o,c,m=self.ff_up.forward(n,store,Config.FFN_DROPOUT if training else 0.0,training); pre.append(o); uc.append(c); um2.append(m)
        post=[gelu_arr(p) for p in pre]
        fo,dc,dm=[],[],[]
        for a in post: o,c,m=self.ff_down.forward(a,store,0.0,training); fo.append(o); dc.append(c); dm.append(m)
        x=[array.array('f',[r2[i][j]+fo[i][j] for j in range(d_model)]) for i in range(seq_len)]
        if self.trainable:
            caches={'ln1_caches':ln1_caches,'q_caches':qc,'k_caches':kc,'v_caches':vc,'o_caches':oc,
                    'ln2_caches':c2,'up_caches':uc,'down_caches':dc,'Q_heads':Qh,'K_heads':Kh,'V_heads':Vh,
                    'all_probs':ap,'pre_gelu':pre,'positions':pos_local,'causal_mask':am,
                    'q_masks':qm,'k_masks':km,'v_masks':vm,'o_masks':om,'up_masks':um2,'down_masks':dm}
        return x, caches
    
    def backward(self, grad_output_list, caches, store, grads):
    if not self.trainable or not caches: return [array.array('f',[0.0]*self.d_model) for _ in grad_output_list]
    c=caches; seq_len=len(grad_output_list); d_model=self.d_model; d_head=self.d_head
    n_heads=self.n_heads; scale=math.sqrt(d_head); positions=c['positions']
    
    gd=[self.ff_down.backward(grad_output_list[i],c['down_caches'][i],c['down_masks'][i],store,grads) for i in range(seq_len)]
    gpost=[array.array('f',[gd[i][k]*gelu_derivative_arr(c['pre_gelu'][i])[k] for k in range(len(gd[i]))]) for i in range(seq_len)]
    gu=[self.ff_up.backward(gpost[i],c['up_caches'][i],c['up_masks'][i],store,grads) for i in range(seq_len)]
    gl2=[self.ln2.backward(gu[i],c['ln2_caches'][i],store,grads) for i in range(seq_len)]
    grad_res2=[array.array('f',[grad_output_list[i][j]+gl2[i][j] for j in range(d_model)]) for i in range(seq_len)]
    
    go=[self.o_proj.backward(grad_res2[i],c['o_caches'][i],c['o_masks'][i],store,grads) for i in range(seq_len)]
    gQ=[array.array('f',[0.0]*d_model) for _ in range(seq_len)]
    gK=[array.array('f',[0.0]*d_model) for _ in range(seq_len)]
    gV=[array.array('f',[0.0]*d_model) for _ in range(seq_len)]
    for h in range(n_heads):
        s,e=h*d_head,(h+1)*d_head; Vh=c['V_heads'][h]; ph=c['all_probs'][h]; mh=c['causal_mask'][h]
        Kh_h=c['K_heads'][h]; Qh_h=c['Q_heads'][h]; total_len=len(Vh)
        for i in range(seq_len):
            gh=array.array('f',[go[i][s+k] for k in range(d_head)]); probs=ph[i]; mi=mh[i]
            gp=[0.0]*total_len
            for j in range(total_len):
                if mi[j]:
                    acc=0.0
                    for k in range(d_head): acc+=float(gh[k])*float(Vh[j][k])
                    gp[j]=acc
            gs=softmax_backward(probs,mi,gp,scale)
            for j in range(total_len):
                if not mi[j] or abs(gs[j])<1e-30: continue
                Kh_j=Kh_h[j]; Qh_i=Qh_h[i]
                for k in range(d_head): gQ[i][s+k]+=gs[j]*float(Kh_j[k]); gK[j][s+k]+=gs[j]*float(Qh_i[k])
            for j in range(total_len):
                if not mi[j]: continue
                p=float(probs[j])
                if p<=0: continue
                for k in range(d_head): gV[j][s+k]+=p*float(gh[k])
    
    for i in range(seq_len): gQ[i]=self.rope.inverse_rotate(gQ[i],positions[i])
    for j in range(seq_len): gK[j]=self.rope.inverse_rotate(gK[j],positions[j])
    gq=[self.q_proj.backward(gQ[i],c['q_caches'][i],c['q_masks'][i],store,grads) for i in range(seq_len)]
    gk=[self.k_proj.backward(gK[i],c['k_caches'][i],c['k_masks'][i],store,grads) for i in range(seq_len)]
    gv=[self.v_proj.backward(gV[i],c['v_caches'][i],c['v_masks'][i],store,grads) for i in range(seq_len)]
    grad_attn=[array.array('f',[gq[i][j]+gk[i][j]+gv[i][j] for j in range(d_model)]) for i in range(seq_len)]
    gl1=[self.ln1.backward(grad_attn[i],c['ln1_caches'][i],store,grads) for i in range(seq_len)]
    return [array.array('f',[grad_res2[i][j]+gl1[i][j] for j in range(d_model)]) for i in range(seq_len)]

# ====================================================================
# TOKENIZER
# ====================================================================
class BPETokenizer:
    def __init__(self):
        self.vocab_size=Config.VOCAB_SIZE; self.merges={}; self.merge_rank={}
        self.vocab={}; self.reverse_vocab={}
        self.st={'<pad>':0,'<unk>':1,'<s>':2,'</s>':3}; self.next_id=4; self._init_vocab()
    def _init_vocab(self):
        for i in range(256): tid=self.next_id; self.next_id+=1; self.vocab[tid]=bytes([i]); self.reverse_vocab[bytes([i])]=tid
        for tok,tid in self.st.items(): self.vocab[tid]=tok.encode(); self.reverse_vocab[tok.encode()]=tid
    def train(self, texts, num_merges=3000):
        pc=Counter()
        for text in texts[:1000]:
            raw=text.encode('utf-8',errors='replace')
            toks=[self.reverse_vocab.get(bytes([t]),1) for t in raw]
            for i in range(len(toks)-1): pc[(toks[i],toks[i+1])]+=1
        ranked=sorted(pc.items(),key=lambda x:-x[1])
        for rank,(pair,_) in enumerate(ranked[:num_merges]):
            if pair in self.merges: continue
            tid=self.next_id; self.next_id+=1
            if tid>=self.vocab_size: break
            self.merges[pair]=tid; self.merge_rank[pair]=rank
            nb=self.vocab[pair[0]]+self.vocab[pair[1]]
            self.vocab[tid]=nb; self.reverse_vocab[nb]=tid
    def _apply_merges_heap(self, tokens):
        merged=list(tokens); n=len(merged); heap=[]
        for i in range(n-1):
            pair=(merged[i],merged[i+1])
            if pair in self.merge_rank: heapq.heappush(heap,(self.merge_rank[pair],i,pair))
        while heap:
            rank,idx,pair=heapq.heappop(heap)
            if idx>=len(merged)-1: continue
            if (merged[idx],merged[idx+1])!=pair: continue
            merged[idx]=self.merges[pair]; merged.pop(idx+1); n-=1
            for nb in [idx-1,idx]:
                if 0<=nb<n-1:
                    np=(merged[nb],merged[nb+1])
                    if np in self.merge_rank: heapq.heappush(heap,(self.merge_rank[np],nb,np))
        return merged
    def encode(self, text):
        try: raw=text.encode('utf-8')
        except: raw=text.encode('utf-8',errors='replace')
        toks=[self.reverse_vocab.get(bytes([t]),1) for t in raw]
        return tuple([self.st['<s>']]+self._apply_merges_heap(toks))
    def decode(self, tokens):
        r=b''
        for t in tokens:
            if t in self.vocab and t>=4: r+=self.vocab[t]
            elif t>=4: r+=b'?'
        return r.decode('utf-8',errors='replace').strip()

# ====================================================================
# MODELO COM STAGED TRAINING
# ====================================================================
class QICCRLLM:
    def __init__(self):
        self.tokenizer = BPETokenizer()
        # Cálculo do espaço total baseado no estágio FINAL para evitar realocação de memória física
        d_final = Config.S2_D_MODEL
        df_final = Config.S2_D_FF
        nl_final = Config.S2_N_LAYERS
        
        # O total deve considerar o layout do estágio 2 desde o início
        total = Config.VOCAB_SIZE * d_final + \
                nl_final * (4 * d_final * (d_final + 1) + df_final * (d_final + 1) + d_final * (df_final + 1) + 4 * d_final) + \
                2 * d_final + 1
        
        self.store = WeightStore(total)
        self.grads = array.array('f', [0.0] * total)
        self.opt = AdamOptimizer(total)
        self.alloc = WeightAllocator(total)
        
        # O offset do token embedding é fixo no início
        self.tok_off = self.alloc.alloc(Config.VOCAB_SIZE * d_final, "tok_embed")
        self.stage = 1
        self._build_stage1()
        
        # KV Cache precisa ser compatível com as dimensões atuais
        self.kv = KVCache(Config.S1_N_LAYERS, Config.S1_N_HEADS, Config.S1_D_MODEL // Config.S1_N_HEADS, Config.KV_MAX_SEQ)
        self.step = 0
        self.gpos = 0
    
    def _build_stage1(self):
        d=Config.S1_D_MODEL; nh=Config.S1_N_HEADS; nl=Config.S1_N_LAYERS; df=Config.S1_D_FF
        self.d_model=d; self.n_heads=nh; self.n_layers=nl; self.d_ff=df
        self.layers=[TransformerBlock(i,self.alloc,self.store,d,nh,df,True) for i in range(nl)]
        self.ln_off=self.alloc.alloc(2*d,"ln_final"); self.ln=LayerNorm(d,self.ln_off,"ln_final",True)
        self.ls_off=self.alloc.alloc(1,"logit_scale"); self.store.write_fp32(self.ls_off,1.0/math.sqrt(d))
        self._init_weights()
    
    def _init_weights(self):
        d=self.d_model
        for i in range(Config.VOCAB_SIZE):
            for j in range(d): self.store.write_fp32(self.tok_off+i*d+j,random.gauss(0,0.02))
        for layer in self.layers:
            for proj in [layer.q_proj,layer.k_proj,layer.v_proj,layer.o_proj,layer.ff_up,layer.ff_down]:
                self._xavier(proj.offset,proj.in_f,proj.out_f)
            for off in [layer.ln1.offset,layer.ln1.offset+d,layer.ln2.offset,layer.ln2.offset+d]:
                is_g=off in [layer.ln1.offset,layer.ln2.offset]
                for i in range(d): self.store.write_fp32(off+i,1.0 if is_g else 0.0)
        for off in [self.ln_off,self.ln_off+d]:
            for i in range(d): self.store.write_fp32(off+i,1.0 if off==self.ln_off else 0.0)
    
    def _xavier(self,off,in_f,out_f):
        std=math.sqrt(2.0/(in_f+out_f))
        for i in range(out_f):
            base=off+i*(in_f+1)
            for j in range(in_f): self.store.write_fp32(base+j,random.gauss(0,std))
            self.store.write_fp32(base+in_f,0.0)
    
    def _noam_lr(self,step):
        d=self.d_model; warmup=Config.NOAM_WARMUP; step=max(1,step)
        return d**(-0.5)*min(step**(-0.5),step*warmup**(-1.5))
    
    def _interpolate_weights(self, old_off, new_off, old_in, old_out, new_in, new_out):
        """
        Copia pesos de uma matriz antiga para uma nova, preservando o aprendizado.
        Lida com o bias (última coluna) corretamente.
        """
        for i in range(min(old_out, new_out)):
            # Offset de matrizes lineares (weights + bias)
            old_base = old_off + i * (old_in + 1)
            new_base = new_off + i * (new_in + 1)
            
            # Copia pesos das colunas
            for j in range(min(old_in, new_in)):
                self.store.write_fp32(new_base + j, self.store.read_fp32(old_base + j))
            
            # Copia o BIAS (que fica no final da linha)
            self.store.write_fp32(new_base + new_in, self.store.read_fp32(old_base + old_in))
         
    def expand_to_stage2(self):
        if self.stage >= 2: return
        print("🚀 Expandindo para Estágio 2 (Growth: Width & Depth)...")
        
        old_d, new_d = Config.S1_D_MODEL, Config.S2_D_MODEL
        old_df, new_df = Config.S1_D_FF, Config.S2_D_FF
        
        # 1. Backup de referências
        old_tok_off = self.tok_off
        old_layers = self.layers[:]
        old_ln_off = self.ln_off

        # 2. Re-alocação lógica (O WeightStore continua o mesmo)
        self.alloc = WeightAllocator(self.store.total_params)
        self.tok_off = self.alloc.alloc(Config.VOCAB_SIZE * new_d, "tok_embed")
        
        # 3. Interpolação correta do Embedding
        # Diferente de matrizes lineares, o embedding é [Vocab x Dim] sem bias por linha
        for i in range(Config.VOCAB_SIZE):
            for j in range(min(old_d, new_d)):
                val = self.store.read_fp32(old_tok_off + i * old_d + j)
                self.store.write_fp32(self.tok_off + i * new_d + j, val)

        # 4. Construção das novas camadas
        self.layers = []
        for i in range(Config.S2_N_LAYERS):
            blk = TransformerBlock(i, self.alloc, self.store, new_d, Config.S2_N_HEADS, new_df, True)
            self.layers.append(blk)

        # 5. Width Growth (Camada 0 antiga -> Camada 0 nova)
        o_blk, n_blk = old_layers[0], self.layers[0]
        projs = [(o_blk.q_proj, n_blk.q_proj), (o_blk.k_proj, n_blk.k_proj), 
                 (o_blk.v_proj, n_blk.v_proj), (o_blk.o_proj, n_blk.o_proj),
                 (o_blk.ff_up, n_blk.ff_up), (o_blk.ff_down, n_blk.ff_down)]
        
        for op, np in projs:
            self._interpolate_weights(op.offset, np.offset, op.in_f, op.out_f, np.in_f, np.out_f)
            
        # 6. Depth Stacking (Se subimos de 1 para 2 camadas, a L1 recebe pesos da L0 expandida)
        if len(self.layers) > 1:
            l0, l1 = self.layers[0], self.layers[1]
            # Copiar bytes brutos da L0 expandida para a L1
            for p0, p1 in [(l0.q_proj, l1.q_proj), (l0.k_proj, l1.k_proj), (l0.v_proj, l1.v_proj),
                           (l0.o_proj, l1.o_proj), (l0.ff_up, l1.ff_up), (l0.ff_down, l1.ff_down)]:
                for k in range(p0.W_size):
                    self.store.write_fp32(p1.offset + k, self.store.read_fp32(p0.offset + k))

        # 7. Finalização
        self.ln_off = self.alloc.alloc(2 * new_d, "ln_final")
        self.ln = LayerNorm(new_d, self.ln_off, "ln_final", True)
        self.ls_off = self.alloc.alloc(1, "logit_scale")
        self.store.write_fp32(self.ls_off, 1.0 / math.sqrt(new_d))
        
        # Reset do Cache para novas dimensões
        self.d_model = new_d
        self.kv = KVCache(Config.S2_N_LAYERS, Config.S2_N_HEADS, new_d // Config.S2_N_HEADS, Config.KV_MAX_SEQ)
        self.stage = 2
        
        # Resetar momentos do Adam apenas para os parâmetros que mudaram drasticamente
        self.opt = AdamOptimizer(self.store.total_params) 
        print("✅ Expansão concluída.")
    
    def _forward(self,tokens,use_kv=False,positions=None,training=False):
        d=self.d_model; seq=tokens
        x=[self.store.read_vector_fp32(self.tok_off+tid*d,d) for tid in seq]
        kv=self.kv if use_kv else None; cl=[]
        for layer in self.layers: x,c=layer.forward(x,self.store,kv_cache=kv,positions=positions,training=training); cl.append(c)
        final,lnc=self.ln.forward(x[-1],self.store)
        ls=max(Config.LOGIT_SCALE_MIN,min(Config.LOGIT_SCALE_MAX,self.store.read_fp32(self.ls_off)))
        logits=array.array('f',[ls*sum(self.store.read_fp32(self.tok_off+t*d+j)*final[j] for j in range(d)) for t in range(Config.VOCAB_SIZE)])
        return logits,cl,lnc
    
    def prefill(self,prompt):
        self.kv.clear(); self.gpos=0; pos=list(range(len(prompt)))
        logits,_,_=self._forward(tuple(prompt),use_kv=True,positions=pos,training=False)
        self.gpos=len(prompt); return logits
    
    def decode_step(self,tid):
        logits,_,_=self._forward((tid,),use_kv=True,positions=[self.gpos],training=False)
        self.gpos+=1; return logits
    
    def train_step(self,contexts,targets):
        d=self.d_model; bs=len(contexts); total_loss=0.0
        for ctx,tgt in zip(contexts,targets):
            seq=ctx[-Config.MAX_SEQ:]; sl=len(seq); pos=list(range(sl))
            x=[self.store.read_vector_fp32(self.tok_off+tid*d,d) for tid in seq]; cl=[]
            for layer in self.layers: x,c=layer.forward(x,self.store,kv_cache=None,positions=pos,training=True); cl.append(c)
            final,lnc=self.ln.forward(x[-1],self.store)
            ls=max(Config.LOGIT_SCALE_MIN,min(Config.LOGIT_SCALE_MAX,self.store.read_fp32(self.ls_off)))
            logits=array.array('f',[ls*sum(self.store.read_fp32(self.tok_off+t*d+j)*final[j] for j in range(d)) for t in range(Config.VOCAB_SIZE)])
            probs=safe_softmax(logits)
            smooth,V=Config.LABEL_SMOOTHING,Config.VOCAB_SIZE
            td=array.array('f',[smooth/V]*V); td[tgt]=(1.0-smooth)+smooth/V
            loss=-sum(td[i]*math.log(max(probs[i],1e-9)) for i in range(V))
            norm=1.0/bs; gl=array.array('f',[(probs[i]-td[i])*norm for i in range(V)])
            gf=array.array('f',[0.0]*d)
            for t in range(V):
                g=gl[t]
                for j in range(d): gf[j]+=self.store.read_fp32(self.tok_off+t*d+j)*g*ls; self.grads[self.tok_off+t*d+j]+=final[j]*g*ls
            self.grads[self.ls_off]+=sum(gl[t]*sum(self.store.read_fp32(self.tok_off+t*d+j)*final[j] for j in range(d)) for t in range(V))
            gln=self.ln.backward(gf,lnc,self.store,self.grads)
            glist=[array.array('f',[0.0]*d) for _ in range(sl)]; glist[-1]=gln
            for idx in range(len(self.layers)-1,-1,-1): glist=self.layers[idx].backward(glist,cl[idx],self.store,self.grads)
            en=norm/max(1,sl)
            for pi,tid in enumerate(seq):
                base=self.tok_off+tid*d
                for j in range(d): self.grads[base+j]+=glist[pi][j]*en
            total_loss+=loss
        avg_loss=total_loss/bs
        trainable=[(self.tok_off,Config.VOCAB_SIZE*d),(self.ln_off,2*d),(self.ls_off,1)]
        for layer in self.layers:
            for proj in [layer.q_proj,layer.k_proj,layer.v_proj,layer.o_proj,layer.ff_up,layer.ff_down]:
                trainable.append((proj.offset,proj.W_size))
            trainable.append((layer.ln1.offset,2*d)); trainable.append((layer.ln2.offset,2*d))
        if self.step%100==0:
            gn=math.sqrt(sum(self.grads[off+i]**2 for off,sz in trainable for i in range(sz)))
            print(f"   [GradNorm] step {self.step}: {gn:.4f} | LR: {self._noam_lr(max(1,self.step)):.8f}")
        self.opt.step(self.store.fp32,self.grads,trainable,self._noam_lr(max(1,self.step)),Config.WEIGHT_DECAY,Config.GRAD_CLIP)
        self.step+=1; return avg_loss
    
    def generate_beam(self,prompt,max_new=80,temp=0.7,bw=None):
    if bw is None: bw=Config.BEAM_WIDTH
    EOS=self.tokenizer.st.get('</s>',3); d=self.d_model; nl=self.n_layers
    pt=list(self.tokenizer.encode(prompt))
    def make_initial():
        cache=KVCache(nl,self.n_heads,d//self.n_heads,Config.KV_MAX_SEQ)
        pos=list(range(len(pt)))
        x=[self.store.read_vector_fp32(self.tok_off+tid*d,d) for tid in pt]
        for layer in self.layers: x,_=layer.forward(x,self.store,kv_cache=cache,positions=pos,training=False)
        final,_=self.ln.forward(x[-1],self.store)
        ls=max(Config.LOGIT_SCALE_MIN,min(Config.LOGIT_SCALE_MAX,self.store.read_fp32(self.ls_off)))
        logits=array.array('f',[ls*sum(self.store.read_fp32(self.tok_off+t*d+j)*final[j] for j in range(d)) for t in range(Config.VOCAB_SIZE)])
        return logits,cache,len(pt)
    first_logits,first_cache,first_pos=make_initial()
    lt=array.array('f',[l/temp for l in first_logits]); probs=safe_softmax(lt)
    topk=sorted(enumerate(probs),key=lambda x:x[1],reverse=True)[:bw]
    beams=[{'tokens':pt+[tid],'log_prob':math.log(max(p,1e-9)),'cache':first_cache.clone(),'pos':first_pos,'finished':tid==EOS} for tid,p in topk]
    for _ in range(max_new-1):
        if all(b['finished'] for b in beams): break
        candidates=[]
        for b in beams:
            if b['finished']: candidates.append(b); continue
            last=b['tokens'][-1]; cp=b['pos']
            x=[self.store.read_vector_fp32(self.tok_off+last*d,d)]
            for layer in self.layers: x,_=layer.forward(x,self.store,kv_cache=b['cache'],positions=[cp],training=False)
            final,_=self.ln.forward(x[-1],self.store)
            ls=max(Config.LOGIT_SCALE_MIN,min(Config.LOGIT_SCALE_MAX,self.store.read_fp32(self.ls_off)))
            logits=array.array('f',[ls*sum(self.store.read_fp32(self.tok_off+t*d+j)*final[j] for j in range(d)) for t in range(Config.VOCAB_SIZE)])
            lt=array.array('f',[l/temp for l in logits]); probs=safe_softmax(lt)
            for tid,p in sorted(enumerate(probs),key=lambda x:x[1],reverse=True)[:bw]:
                nc=b['cache'].clone()
                candidates.append({'tokens':b['tokens']+[tid],'log_prob':b['log_prob']+math.log(max(p,1e-9)),'cache':nc,'pos':cp+1,'finished':tid==EOS})
        beams=sorted(candidates,key=lambda x:x['log_prob'],reverse=True)[:bw]
    return self.tokenizer.decode(beams[0]['tokens'][len(pt):])
    
    def save(self,path="qiccr_v7"):
        with open(path+"_fp32.bin",'wb') as f: f.write(self.store.fp32.tobytes())
        self.opt.save(path+"_optim.json.gz")
        meta={'step':self.step,'stage':self.stage,'merges':{f"{k[0]},{k[1]}":v for k,v in self.tokenizer.merges.items()},
              'merge_rank':{f"{k[0]},{k[1]}":v for k,v in self.tokenizer.merge_rank.items()},
              'vocab':{str(k):list(v) for k,v in self.tokenizer.vocab.items()},'next_id':self.tokenizer.next_id}
        with gzip.open(path+"_meta.json.gz",'wt') as f: json.dump(meta,f)
    
    def load(self,path="qiccr_v7"):
        if not os.path.exists(path+"_fp32.bin"): return False
        with open(path+"_fp32.bin",'rb') as f: self.store.fp32=array.array('f'); self.store.fp32.frombytes(f.read())
        self.opt.load(path+"_optim.json.gz")
        with gzip.open(path+"_meta.json.gz",'rt') as f: meta=json.load(f)
        self.step=meta['step']; self.stage=meta.get('stage',2)
        self.tokenizer.merges={tuple(map(int,k.split(','))):v for k,v in meta['merges'].items()}
        self.tokenizer.merge_rank={tuple(map(int,k.split(','))):v for k,v in meta.get('merge_rank',{}).items()}
        self.tokenizer.vocab={int(k):bytes(v) for k,v in meta['vocab'].items()}
        self.tokenizer.reverse_vocab={v:k for k,v in self.tokenizer.vocab.items()}
        self.tokenizer.next_id=meta.get('next_id',4)
        return True

# ====================================================================
# TREINAMENTO COM ESTÁGIOS (REFATORADO)
# ====================================================================
def train_model(model, file="treino.txt"):
    if not os.path.exists(file): 
        print("❌ Arquivo não encontrado!"); return
    
    with open(file, 'r', encoding='utf-8') as f: 
        text = f.read().strip()
    
    print(f"📚 {len(text):,} caracteres")
    model.tokenizer.train([text])
    toks = list(model.tokenizer.encode(text))
    print(f"🔢 {len(toks):,} tokens")
    
    # Definição dos estágios para iterar
    stages = [
        {"name": "ESTÁGIO 1: Modelo Base (1 camada, d=64)", "steps": Config.S1_STEPS, "expand": False},
        {"name": "ESTÁGIO 2: Expansão (2 camadas, d=128)", "steps": Config.S2_STEPS, "expand": True},
        {"name": "ESTÁGIO 3: Fine-tuning Final", "steps": Config.S3_STEPS, "expand": False}
    ]

    best_loss = float('inf')

    for idx, stage in enumerate(stages):
        if stage["expand"]:
            model.expand_to_stage2()
        
        print(f"\n🔥 {stage['name']}")
        steps = min(stage["steps"], len(toks) - Config.TRAIN_WINDOW_MAX - 2)
        total_loss = 0.0
        
        # Epoch única por estágio para simplificar
        print(f"🏋️ Treinando ({steps} passos, batch={Config.BATCH_SIZE})...")
        
        for i in range(steps):
            bctx, btgt = [], []
            for _ in range(Config.BATCH_SIZE):
                # Janela dinâmica para robustez
                w = random.randint(Config.TRAIN_WINDOW_MIN, Config.TRAIN_WINDOW_MAX)
                s = random.randint(0, max(0, len(toks) - w - 2))
                bctx.append(toks[s:s+w])
                btgt.append(toks[s+w])
            
            loss = model.train_step(bctx, btgt)
            total_loss += loss
            
            if (i + 1) % 500 == 0 or i == steps - 1:
                avg = total_loss / (i + 1)
                print(f"   Passo {i+1:5d}/{steps} | Loss: {avg:.4f}")

        # Lógica de salvamento ao final de cada estágio importante
        current_avg = total_loss / steps
        if current_avg < best_loss:
            best_loss = current_avg
            model.save("qiccr_v7_best")
        model.save("qiccr_v7_latest")

    print("\n🏁 Treinamento completo e modelo salvo!")

# ====================================================================
# INTERFACE E EXECUÇÃO
# ====================================================================
def interactive_chat(model):
    print("\n🚀 QICCR-LLM v7.0 | 'sair' | 'reset' | 'beam'\n")
    use_beam = False
    while True:
        try:
            q = input("🧑 Você: ").strip()
            if not q: continue
            if q.lower() in ('sair', 'exit', 'quit'): break
            if q.lower() == 'reset': 
                model.kv.clear(); model.gpos = 0
                print("✅ Cache resetado.\n"); continue
            if q.lower() == 'beam': 
                use_beam = not use_beam
                print(f"✅ Beam Search: {'LIGADO' if use_beam else 'DESLIGADO'}\n"); continue
            
            # Nota: Certifique-se que o método generate existe na classe QICCRLLM
            res = model.generate(q, max_new=80, temp=0.7, use_beam=use_beam)
            print(f"🐶 Qiccr: {res}\n")
        except KeyboardInterrupt: 
            break

if __name__ == "__main__":
    random.seed(42)
    model = QICCRLLM()
    
    if "--train" in sys.argv:
        train_model(model)
    else:
        # Tenta carregar o melhor modelo primeiro, depois o último
        loaded = False
        for tag in ("qiccr_v7_best", "qiccr_v7_latest"):
            if os.path.exists(tag + "_fp32.bin") or os.path.exists(tag + ".json"): # ajuste conforme seu model.save
                try:
                    loaded = model.load(tag)
                    if loaded: 
                        print(f"✅ Modelo carregado: {tag}")
                        break
                except: continue
        
        if not loaded: 
            print("⚠️ Sem checkpoint encontrado. Iniciando chat com pesos aleatórios ou use --train")
        
        interactive_chat(model)
