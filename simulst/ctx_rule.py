"""Context rule of the streaming policy.

When the audio window slides, target tokens committed before the window start are kept in the prompt as context:
at most the last `limit` tokens, starting at a sentence beginning, i.e. the earliest position i in
[end - limit, end) such that ids[i - 1] is a sentence-final token (its text, after stripping closing quotes,
brackets and whitespace, ends with 。！？!?). If there is no such position, the context starts at end - limit,
skipping leading tokens that decode to an incomplete character (U+FFFD). Everything is kept when fewer than
`limit` tokens exist; limit <= 0 keeps nothing.
"""
_END = frozenset('。！？!?')
_CLOSE = '”’"\'）)》」』】]} \t\r\n'
_CACHE = {}


def token_sets(decode, n_regular):
    """Return (sentence-final token ids, incomplete-character token ids).

    decode(ids) -> str; n_regular = number of regular tokens (= smallest special-token id).
    """
    if n_regular not in _CACHE:
        ends, partial = set(), set()
        for i in range(n_regular):
            s = decode([i])
            if '\ufffd' in s:
                partial.add(i)
            t = s.rstrip(_CLOSE)
            if t and t[-1] in _END:
                ends.add(i)
        _CACHE[n_regular] = (frozenset(ends), frozenset(partial))
    return _CACHE[n_regular]


def history_start(ids, end, limit, ends, partial):
    """Return h such that ids[h:end] is the context to keep."""
    if limit <= 0 or end <= 0:
        return end
    lo = end - limit
    if lo <= 0:
        return 0
    for i in range(lo, end):
        if ids[i - 1] in ends:
            return i
    while lo < end and ids[lo] in partial:
        lo += 1
    return lo
