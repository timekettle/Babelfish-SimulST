"""Repetition guard: check a step's output before it is committed and cut it where a loop starts.

A model can fall into a loop (the same sentence again and again; mostly with greedy decoding, rarely with beam
search). Once a repeated sentence is committed it becomes part of the prefix and feeds the loop, so the guard looks
at the new tokens of a step *before* they are committed. If one of the signs below fires, the step keeps only the
tokens before the point where the copying starts; the rest is never output and never enters the prefix. Steps
where nothing fires are committed unchanged. The signs only fire on runaway repetition, so that normal output,
including a speaker who repeats a phrase, is left alone.

Signs (token level; `history` = the last HIST committed tokens):
  rep    the new tokens contain a span of >= 3 tokens immediately repeated, and the periodic run around it
         (extending back into the history) holds >= 3 full copies; or the step hit the token limit and contains
         an immediate repetition of any count;
  xrep   history + new tokens end in a periodic run with a unit of >= 3 tokens and >= 3 full copies, and the new
         tokens hold at least one full unit (a sentence repeated across steps, one copy per step);
  lcopy  a run of >= 24 tokens ending in the new tokens already occurred earlier (a passage translated again).
"""
HIST = 192          # committed tokens looked at
LCOPY_M = 24        # minimum length of a verbatim re-translation (`lcopy`)


def immediate_repeats(t, k=3):
    """All (start, unit) with t[s:s+p] == t[s+p:s+2p], p >= k; the earliest start for every unit length."""
    n, out = len(t), []
    for p in range(k, n // 2 + 1):
        for s in range(0, n - 2 * p + 1):
            if t[s:s + p] == t[s + p:s + 2 * p]:
                out.append((s, p))
                break
    return out


def min_period(x, p):
    for q in range(1, p + 1):
        if p % q == 0 and all(x[i] == x[i + q] for i in range(len(x) - q)):
            return q
    return p


def cut_rep(hist, t, reps):
    """Where copying starts: the periodic run is extended back into the history; keep its first unit."""
    seq = list(hist) + list(t)
    H = len(hist)
    best = len(t)
    for s, p in reps:
        q = min_period(t[s:s + 2 * p], p)
        start = H + s
        while start > 0 and seq[start - 1] == seq[start - 1 + q]:
            start -= 1
        best = min(best, max(0, start + q - H))
    return best


def n_copies(hist, t, s, p):
    """Full copies in the periodic run around the immediate repetition (s, p) (unit counted as >= 3 tokens)."""
    seq = list(hist) + list(t)
    H = len(hist)
    q = min_period(t[s:s + 2 * p], p)
    if q < 3:
        q = p
    a = H + s
    while a > 0 and seq[a - 1] == seq[a - 1 + q]:
        a -= 1
    b = H + s + 2 * p
    while b < len(seq) and seq[b] == seq[b - q]:
        b += 1
    return (b - a) // q


def xrep_cut(hist, t, max_p=64):
    if not t:
        return None
    seq = list(hist) + list(t)
    n, H = len(seq), len(hist)
    best = None
    for p in range(3, min(max_p, n // 3, len(t)) + 1):     # the step itself adds at least one full unit
        if seq[-1] != seq[-1 - p]:
            continue
        a = n - p
        while a > 0 and seq[a - 1] == seq[a - 1 + p]:
            a -= 1
        if (n - a) // p >= 3:
            c = max(0, a + p - H)
            best = c if best is None else min(best, c)
    return best


def lcopy_cut(hist, t, m=LCOPY_M):
    if not t:
        return None
    seq = list(hist) + list(t)
    H = len(hist)
    first = {}
    for e in range(m - 1, len(seq)):
        g = tuple(seq[e - m + 1:e + 1])
        if e >= H and g in first:
            s, s0 = e - m + 1, first[g] - m + 1
            while s > 0 and s0 > 0 and seq[s - 1] == seq[s0 - 1]:
                s, s0 = s - 1, s0 - 1
            return max(0, s - H)
        first.setdefault(g, e)
    return None


def check(committed, new, audio_s=None, hit_max=False):
    """committed: committed token ids; new: this step's tokens (not committed yet); hit_max: the step stopped at
    the token limit (audio_s is accepted for compatibility and not used). Returns (tokens to keep, rule) or
    (len(new), None)."""
    if not new:
        return 0, None
    t = list(new)
    hist = list(committed[-HIST:])
    fired, cuts = [], []
    reps = immediate_repeats(t, 3)
    if reps and not hit_max:
        reps = [(s, p) for s, p in reps if n_copies(hist, t, s, p) >= 3]
    if reps:
        fired.append('rep')
        cuts.append(cut_rep(hist, t, reps))
    c = xrep_cut(hist, t)
    if c is not None:
        fired.append('xrep')
        cuts.append(c)
    c = lcopy_cut(hist, t)
    if c is not None:
        fired.append('lcopy')
        cuts.append(c)
    if not fired:
        return len(t), None
    return min(cuts), '+'.join(fired)
