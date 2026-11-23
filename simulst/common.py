"""Constants shared by every inference path."""
import os

SR = 16000                 # the model takes 16 kHz mono audio
STEP_S = 0.32              # one decision step = 320 ms of audio
STEP_SAMPLES = 5120        # STEP_S * SR
END_MARK = ' [audio complete]'   # appended to the user instruction once the whole input has been received
TARGET_LANG = dict(en2zh='zh', zh2en='en')
DIRECTIONS = tuple(TARGET_LANG)

# (system prompt, user instruction) per direction; the model expects exactly these strings.
PROMPTS = {
    'en2zh': ('You are a professional speech translator. Translate English speech into accurate Chinese. '
              'Output only the Chinese translation.', 'Translate this English speech into Chinese.'),
    'zh2en': ('You are a professional speech translator. Translate Chinese speech into accurate English. '
              'Output only the English translation.', 'Translate this Chinese speech into English.'),
}

# Default decoding settings
DEFAULT_WINDOW_S = 24.0    # audio window
DEFAULT_CTX_TOKENS = 256   # committed target tokens kept as context once they leave the window
DEFAULT_BEAMS = 1          # greedy decoding (with the repetition guard); beam size 2 is optional
DEFAULT_MAX_NEW = 96       # max new tokens per step


def normalize_direction(d):
    """Accept en2zh / en-zh / en_zh (and the zh2en variants)."""
    d = (d or '').strip().lower().replace('-', '2').replace('_', '2')
    if d not in TARGET_LANG:
        raise ValueError(f'unsupported direction {d!r}; use one of {", ".join(DIRECTIONS)}')
    return d


def drop_dist_env():
    """vLLM misreads torch.distributed variables injected by some cluster schedulers."""
    for k in ('MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK'):
        os.environ.pop(k, None)
