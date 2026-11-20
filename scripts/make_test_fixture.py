#!/usr/bin/env python3
"""Dependency-free scalar reference and tiny native-layout checkpoint generator.

Equations follow Transformers PR #51050 at 6173a2f679ff4ed393afc499aefc569da7e68bc1.
This deliberately uses ordinary lists, float64 sums, and growing histories rather
than the GPU implementation's kernels, packed buffers, and ring caches.
Run from any directory; the resulting fixtures are checked in for offline tests.
"""
import json
import math
import random
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'models/inkling/tests/fixtures'
OUT = ROOT / 'tiny'
OUT.mkdir(exist_ok=True)
config = json.loads((ROOT / 'official-config.json').read_text())
c = config['text_config']
c.update(hidden_size=8, num_hidden_layers=8, vocab_size=16, unpadded_vocab_size=15,
         num_attention_heads=2, num_key_value_heads=1, head_dim=4,
         swa_num_attention_heads=2, swa_num_key_value_heads=1, swa_head_dim=4,
         d_rel=2, rel_extent=3, sliding_window_size=3, local_layer_ids=[0, 2, 4, 6],
         sconv_kernel_size=2, n_layers_per_attn_res_block=2, n_routed_experts=3,
         num_experts_per_tok=2, intermediate_size=4, model_max_length=16,
         log_scaling_n_floor=2, qk_l2_clip_threshold=0.7, rel_l2_clip_threshold=0.6,
         loop_config=dict(core_start=2, core_end=6, num_passes=3, gate_rank=3, is_varloop=False))
config = dict(model_type='inkling2_loop', eos_token_id=14, text_config=c)
(OUT / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
rng = random.Random(7219)
weights = {}

def put(name, shape, scale=0.4, center=0.):
    # Round to the actual F32 storage before computing reference results.
    data = [struct.unpack('<f', struct.pack('<f', center + rng.uniform(-scale, scale)))[0]
            for _ in range(math.prod(shape))]
    weights[name] = (shape, data)

for name in ('embed.weight', 'unembed.weight'):
    put(name, [16, 8])
for name in ('embed_norm.weight', 'norm.weight'):
    put(name, [8], .1, 1)
put('attn_res.output_query', [8])
for name in ('middle_cycle_cross_pass_queries', 'middle_cycle_cross_pass_kv_queries'):
    put(name, [8, 8])
for i in range(8):
    p = f'layers.{i}'
    local = i in c['local_layer_ids']
    if i:
        put(f'attn_res.attn_queries.{i-1}', [8] if local else [3, 8])
    put(f'attn_res.mlp_queries.{i}', [8])
    for name, rows, cols in [('wq_du', 8, 8), ('wk_dv', 4, 8), ('wv_dv', 4, 8),
                             ('wr_du', 4, 8), ('wo_ud', 8, 8), ('gate_attn', 8, 8),
                             ('attn_sink_proj', 2, 8)]:
        put(f'{p}.attn.{name}.weight', [rows, cols])
    put(f'{p}.attn.q_l2_clip.scale', [2, 4], .1, 1)
    put(f'{p}.attn.rel_l2_clip.scale', [2, 2], .1, 1)
    put(f'{p}.attn.rel_logits_proj.proj', [2, 3])
    for kv in ('k', 'v'):
        if local:
            put(f'{p}.attn.{kv}_sconv.weight', [4, 1, 2])
        else:
            put(f'{p}.attn.{kv}_sconv.static_w', [2, 4])
            put(f'{p}.attn.{kv}_sconv.weight_proj', [2, 8])
    for name in ('attn_sconv', 'mlp_sconv'):
        put(f'{p}.{name}.weight', [8, 1, 2])
    for name in (('attn_norm', 'mlp_norm') if local else ('q_input_norm', 'k_input_norm', 'v_input_norm', 'mlp_norm')):
        put(f'{p}.{name}.weight', [8], .1, 1)
    put(f'{p}.mlp.gate.weight', [3, 8])
    put(f'{p}.mlp.gate.bias', [3])
    put(f'{p}.mlp.experts.w13_weight', [3, 8, 8])
    put(f'{p}.mlp.experts.w2_weight', [3, 8, 4])
    if 2 <= i < 6:
        for kind in ('attn', 'k', 'v', 'mlp'):
            put(f'{p}.controllers.{kind}.down_weight', [3, 16])
            put(f'{p}.controllers.{kind}.forget_weight', [8, 3])
            put(f'{p}.controllers.{kind}.gain_weight', [8, 3])

header, payload = {}, bytearray()
for name, (shape, data) in weights.items():
    raw = struct.pack('<'+'f'*len(data), *data)
    header['model.llm.'+name] = dict(dtype='F32', shape=shape, data_offsets=[len(payload), len(payload)+len(raw)])
    payload.extend(raw)
encoded = json.dumps(header, separators=(',', ':')).encode()
encoded += b' ' * (-len(encoded) % 8)
(OUT / 'model.safetensors').write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)


def w(name):
    return weights[name][1]


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def mat(name, x):
    shape, a = weights[name]
    return [dot(a[i*shape[1]:(i+1)*shape[1]], x) for i in range(shape[0])]


def norm(x, name=None):
    r = 1 / math.sqrt(dot(x, x)/len(x) + c['rms_norm_eps'])
    return [v*r*g for v, g in zip(x, w(name) if name else [1]*len(x))]


def softmax(a):
    a = [math.exp(x-max(a)) for x in a]
    return [x/sum(a) for x in a]


def read(states, name, offset=0):
    states = [norm(s) for s in states]
    query = w(name)[offset:offset+8]
    probs = softmax([dot(s, query)/math.sqrt(8) for s in states])
    return [sum(p*s[i] for p, s in zip(probs, states)) for i in range(8)]


def message(x, m, prefix):
    low = mat(prefix+'.down_weight', norm(x)+norm(m))
    f = mat(prefix+'.forget_weight', low)
    g = mat(prefix+'.gain_weight', low)
    return [(1-math.tanh(a))*v + math.tanh(b)*u for v, u, a, b in zip(x, m, f, g)]


histories, kv_cache = {}, {}
fp16_cache = False


def conv(x, prefix, visit, slot, dynamic_input=None):
    history = histories.setdefault((visit, slot), [])
    history.append(x)
    dyn = mat(prefix+'.weight_proj', dynamic_input) if dynamic_input is not None else None
    taps = w(prefix+('.static_w' if dyn is not None else '.weight'))
    out = []
    for i, value in enumerate(x):
        for lag in range(min(2, len(history))):
            tap = taps[lag*len(x)+i] + dyn[lag] if dyn is not None else taps[i*2+1-lag]
            value += history[-lag-1][i] * tap
        out.append(value/math.sqrt(5 if dyn is not None else 3))
    return out


def clip(x, size, threshold, scale=None, tau=1):
    x = [a*b for a, b in zip(x, w(scale))] if scale else x
    out = []
    for start in range(0, len(x), size):
        part = x[start:start+size]
        factor = min(1, threshold/(math.sqrt(dot(part, part))+1e-6))*tau
        out += [v*factor for v in part]
    return out


def attention(inputs, i, visit, pos):
    prefix = f'layers.{i}.attn'
    local = i in c['local_layer_ids']
    q = mat(prefix+'.wq_du.weight', inputs[0])
    k = mat(prefix+'.wk_dv.weight', inputs[1])
    v = mat(prefix+'.wv_dv.weight', inputs[2])
    r = mat(prefix+'.wr_du.weight', inputs[0])
    gate = mat(prefix+'.gate_attn.weight', inputs[0])
    sink = mat(prefix+'.attn_sink_proj.weight', inputs[0])
    k = conv(k, prefix+'.k_sconv', visit, 0, None if local else inputs[1])
    v = conv(v, prefix+'.v_sconv', visit, 1, None if local else inputs[2])
    tau = 1 if local else 1 + .1*math.log(max(1, (pos+1)/2))
    q = clip(q, 4, .7, prefix+'.q_l2_clip.scale', tau)
    k = clip(k, 4, .7)
    v = norm(v)
    if fp16_cache:
        k = [struct.unpack('<e', struct.pack('<e', x))[0] for x in k]
        v = [struct.unpack('<e', struct.pack('<e', x))[0] for x in v]
    r = clip(r, 2, .6, prefix+'.rel_l2_clip.scale', tau)
    history = kv_cache.setdefault(visit, [])
    history.append((k, v))
    first = max(0, len(history)-3) if local else 0
    bank = w(prefix+'.rel_logits_proj.proj')
    out = []
    for head in range(2):
        scores = []
        for key_pos, (key, _) in enumerate(history[first:], first):
            dist = pos-key_pos
            bias = sum(r[head*2+j]*bank[j*3+dist] for j in range(2)) if dist < 3 else 0
            scores.append(dot(q[head*4:head*4+4], key)/2 + bias)
        probs = softmax(scores+[sink[head]])[:-1]
        out += [sum(p*value[channel] for p, (_, value) in zip(probs, history[first:])) for channel in range(4)]
    out = [v/(1+math.exp(-g)) for v, g in zip(out, gate)]
    return mat(prefix+'.wo_ud.weight', out)


def moe(x, i):
    prefix = f'layers.{i}.mlp'
    scores = [1/(1+math.exp(-v)) for v in mat(prefix+'.gate.weight', x)]
    rank = [a+b for a, b in zip(scores, w(prefix+'.gate.bias'))]
    chosen = sorted(range(3), key=lambda i: rank[i], reverse=True)[:2]
    denom = sum(scores[e] for e in chosen)
    result = [0.]*8
    for e in chosen:
        gu = w(prefix+'.experts.w13_weight')[e*64:(e+1)*64]
        gu = [dot(gu[j*8:(j+1)*8], x) for j in range(8)]
        act = [4*math.tanh(g/4)/(1+math.exp(-g))*25*math.tanh(u/25) for g, u in zip(gu[::2], gu[1::2])]
        down = w(prefix+'.experts.w2_weight')[e*32:(e+1)*32]
        for j in range(8):
            result[j] += dot(down[j*4:(j+1)*4], act)*scores[e]/denom*8
    return result


def layer(hidden, blocks, previous, i, visit, pos):
    prefix = f'layers.{i}'
    local = i in c['local_layer_ids']
    inputs = []
    for j, kind in enumerate(('attn', 'k', 'v')):
        x = hidden if i == 0 else read(blocks+[hidden], f'attn_res.attn_queries.{i-1}', 0 if local else j*8)
        if previous is not None:
            bank = 'middle_cycle_cross_pass_queries' if j == 0 else 'middle_cycle_cross_pass_kv_queries'
            m = read(previous, bank, ((i-2)*2+(1 if j==2 else 0))*8)
            x = message(x, m, prefix+'.controllers.'+kind)
        inputs.append(norm(x, prefix+('.attn_norm.weight' if local else '.'+('q', 'k', 'v')[j]+'_input_norm.weight')))
    attn = attention(inputs, i, visit, pos)
    residual = conv(attn, prefix+'.attn_sconv', visit, 2)
    if i % 2 == 0:
        blocks.append(hidden)
    else:
        residual = [a+b for a, b in zip(hidden, residual)]
    x = read(blocks+[residual], f'attn_res.mlp_queries.{i}')
    if previous is not None:
        m = read(previous, 'middle_cycle_cross_pass_queries', ((i-2)*2+1)*8)
        x = message(x, m, prefix+'.controllers.mlp')
    x = moe(norm(x, prefix+'.mlp_norm.weight'), i)
    x = conv(x, prefix+'.mlp_sconv', visit, 3)
    return [a+b for a, b in zip(residual, x)]


def step(token, pos):
    hidden = norm(w('embed.weight')[token*8:(token+1)*8], 'embed_norm.weight')
    blocks, visit = [], 0
    for i in range(2):
        hidden = layer(hidden, blocks, None, i, visit, pos)
        visit += 1
    prehidden, preblocks = hidden, blocks.copy()
    previous = None
    for loop in range(3):
        if loop:
            previous = blocks[len(preblocks)+1:] + [hidden]
            hidden, blocks = prehidden, preblocks.copy()
        for i in range(2, 6):
            hidden = layer(hidden, blocks, previous, i, visit, pos)
            visit += 1
    for i in range(6, 8):
        hidden = layer(hidden, blocks, None, i, visit, pos)
        visit += 1
    hidden = norm(read(blocks+[hidden], 'attn_res.output_query'), 'norm.weight')
    return [30*math.tanh(v/30) for v in mat('unembed.weight', hidden)[:15]]

ids = [1, 3, 2, 5, 4, 1, 7]
for fp16_cache, filename in [(False, 'expected.json'), (True, 'expected-f16.json')]:
    histories.clear()
    kv_cache.clear()
    expected = dict(tokens=ids, logits=[step(token, pos) for pos, token in enumerate(ids)])
    (OUT / filename).write_text(json.dumps(expected, indent=2)+'\n')
print('Wrote', len(weights), 'tensors and', len(ids), 'reference decoding steps to', OUT)
