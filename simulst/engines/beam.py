"""Beam search shared by the vLLM and llama.cpp engines.

beam_search() is a generator: it yields the prefixes to extend by one token and receives, for each of them, the
top-2nb (token id, log P) pairs of the next token, so any backend that can score one next token can drive it.
"""
IM_END = 151645     # <|im_end|>
NEG = -1.0e9


def split_end(ids):
    ids = list(ids)
    ended = bool(ids) and ids[-1] == IM_END
    return (ids[:-1] if ended else ids), ended


def beam_search(n_req, nb, max_new, length_penalty=1.0):
    """Beam search (num_beams = nb, early stopping) as a generator, so that sync and async engines share it.

    Yields jobs [(request index, prefix ids)] to extend by one token; receives, for each job, the top-2nb
    (token id, log P) pairs of the next token. At every position the 2nb best continuations of all live beams are
    ranked; a continuation that ends (<|im_end|>, or the token limit) is finished only if it ranks among the top nb,
    with score = cumulative log P / length ** length_penalty. Stops when nb hypotheses are finished. Returns
    [(suffix, ended)] with the best finished hypothesis per request.
    """
    K2 = 2 * nb
    st = [dict(run=[([], 0.0)], fin=[], t=0, done=False) for _ in range(n_req)]
    while True:
        jobs = [(i, b) for i, x in enumerate(st) if not x['done'] for b, (_, sc) in enumerate(x['run']) if sc > NEG / 2]
        if not jobs:
            break
        tops = yield [(i, st[i]['run'][b][0]) for i, b in jobs]
        cand = {}
        for (i, b), top in zip(jobs, tops):
            sc = st[i]['run'][b][1]
            cand.setdefault(i, []).extend((sc + lp, b, t) for t, lp in top[:K2])
        for i, cs in cand.items():
            x = st[i]
            cs = sorted(cs, key=lambda c: (-c[0], c[1], c[2]))[:K2]
            last = x['t'] + 1 >= max_new
            hit = [t == IM_END or last for _, _, t in cs]
            order = sorted(range(len(cs)), key=lambda j: -(cs[j][0] + (NEG if hit[j] else 0.0)))[:nb]
            L = x['t'] + 1
            for j in range(min(nb, len(cs))):   # only continuations ranked in the top nb can finish
                if hit[j]:
                    x['fin'].append((x['run'][cs[j][1]][0] + [cs[j][2]], cs[j][0] / (L ** length_penalty)))
            x['fin'] = sorted(x['fin'], key=lambda f: -f[1])[:nb]
            x['run'] = [(x['run'][cs[j][1]][0] + [cs[j][2]], cs[j][0] + (NEG if hit[j] else 0.0)) for j in order]
            x['t'] += 1
            if len(x['fin']) >= nb or all(hit) or not any(sc > NEG / 2 for _, sc in x['run']):
                x['done'] = True
    out = []
    for x in st:
        best = x['fin'][0][0] if x['fin'] else (x['run'][0][0] if x['run'] else [])
        out.append(split_end(best))
    return out


def top_k(logprobs, k):
    """vLLM logprobs dict {token: Logprob} -> the k best (token, log P), best first."""
    return [(t, v.logprob) for t, v in sorted(logprobs.items(), key=lambda kv: -kv[1].logprob)[:k]]
