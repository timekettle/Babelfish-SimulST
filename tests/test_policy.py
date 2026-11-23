import numpy as np

from simulst.common import SR, STEP_SAMPLES as STEP
from simulst.policy import Policy, Stream

SPECIAL = {999}


def decode(ids):
    return ''.join(chr(0x4e00 + i) for i in ids if i not in SPECIAL)


def make(seconds, beams=2, **kw):
    p = Policy(**kw)
    p.ctx_sets = (frozenset({50}), frozenset())     # token 50 ends a sentence
    return Stream(p, decode, 'test', 'en2zh', wave=np.zeros(int(seconds * SR), np.float32), beams=beams)


def test_blocks_and_requests():
    s = make(1.0)
    assert s.blocks == 4                              # 0.32 + 0.32 + 0.32 + 0.04 s
    r = s.request(1, final=False)
    assert len(r.audio) == STEP and r.anchor == [] and not r.mark
    assert s.request(4, final=False) is not None      # whole window is >= 320 ms
    assert s.request(4, final=True).mark              # end marker on the extra call


def test_commit_and_window_crop_keep_context():
    s = make(40.0, window=24.0, ctx_tokens=3)
    s.commit([1, 2, 50, 3, 4], ended=True, k=10, special=SPECIAL)    # committed at 3.2 s
    s.commit([60, 61], ended=True, k=20, special=SPECIAL)            # committed at 6.4 s
    assert s.anchor == [1, 2, 50, 3, 4, 60, 61]
    s.commit([], ended=True, k=90, special=SPECIAL)                  # 28.8 s: the window starts at 4.8 s
    # tokens committed inside the window stay; before it, at most 3 tokens of context starting after a sentence end
    assert s.anchor == [3, 4, 60, 61]
    assert s.ws == 90 * STEP - 24 * SR


def test_special_tokens_are_rejected():
    s = make(5.0)
    s.commit([1, 999], ended=True, k=5, special=SPECIAL)
    assert s.committed == [] and s.invalid == 1


def test_guard_default_on():
    assert make(5.0, beams=1).guard and make(5.0, beams=2).guard
    assert not make(5.0, beams=1, loop_guard='off').guard


def test_guard_cuts_a_loop_before_commit():
    s = make(30.0, beams=1)
    unit = [7, 8, 9, 10, 11]
    s.commit([1, 2] + unit * 4, ended=False, k=40, special=SPECIAL)
    assert s.committed == [1, 2] + unit and s.loop_guards == 1


def test_guard_off_keeps_everything():
    s = make(30.0, beams=1, loop_guard='off')
    unit = [7, 8, 9, 10, 11]
    s.commit([1, 2] + unit * 4, ended=False, k=40, special=SPECIAL)
    assert len(s.committed) == 22 and s.loop_guards == 0


def test_display_times():
    s = make(5.0)
    s.commit([1, 2], ended=True, k=3, special=SPECIAL)
    s.commit([3], ended=True, k=6, special=SPECIAL)
    r = s.result()
    assert r['hypothesis'] == decode([1, 2, 3])
    assert r['delays_ms'] == [960.0, 960.0, 1920.0]
