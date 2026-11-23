import math

from simulst.engines.beam import IM_END, beam_search

# toy model: greedy takes token 1 first (p 0.6) and then needs two more uncertain tokens; token 2 (p 0.4) is followed
# by a confident end, which is better per token (length penalty 1.0)
TABLE = {
    (): [(1, 0.6), (2, 0.4), (7, 1e-3), (8, 1e-3)],
    (1,): [(3, 0.36), (4, 0.34), (IM_END, 0.30), (9, 1e-3)],
    (2,): [(IM_END, 0.9), (6, 0.05), (7, 1e-3), (8, 1e-3)],
    (1, 3): [(IM_END, 0.9), (5, 0.05), (7, 1e-3), (8, 1e-3)],
    (1, 4): [(IM_END, 0.5), (5, 0.4), (7, 1e-3), (8, 1e-3)],
}


def model(prefix):
    return [(t, math.log(p)) for t, p in TABLE[tuple(prefix)]]


def run(nb, max_new=10):
    g = beam_search(1, nb, max_new)
    jobs = next(g)
    while True:
        try:
            jobs = g.send([model(prefix) for _, prefix in jobs])
        except StopIteration as e:
            return e.value[0]


def test_one_beam_is_greedy():
    assert run(1) == ([1, 3], True)


def test_two_beams_find_the_better_hypothesis():
    assert run(2) == ([2], True)


def test_token_limit():
    suffix, ended = run(1, max_new=1)
    assert suffix == [1] and not ended
