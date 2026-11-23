import numpy as np

from simulst.common import normalize_direction
from simulst.server.app import Resampler


def test_s16le_partial_frames():
    rs = Resampler(16000, 1, 's16le')
    pcm = (np.arange(10, dtype='<i2') * 100).tobytes()
    a = rs(pcm[:5])                     # 2 samples + half a sample
    b = rs(pcm[5:])
    out = np.concatenate([a, b])
    assert len(out) == 10
    assert np.allclose(out, np.arange(10) * 100 / 32768.0)


def test_f32le_stereo_is_averaged():
    rs = Resampler(16000, 2, 'f32le')
    x = np.array([[0.2, 0.4], [-0.2, 0.0]], dtype='<f4').tobytes()
    assert np.allclose(rs(x), [0.3, -0.1])


def test_resampling_length():
    rs = Resampler(48000, 1, 'f32le')
    x = np.zeros(48000, '<f4').tobytes()
    n = len(rs(x[:96000])) + len(rs(x[96000:], last=True))
    assert abs(n - 16000) <= 2


def test_directions():
    assert normalize_direction('en-zh') == 'en2zh'
    assert normalize_direction('ZH2EN') == 'zh2en'
