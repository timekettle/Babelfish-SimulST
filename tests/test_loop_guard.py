from simulst import loop_guard as lg

HIST = list(range(1000, 1060))          # 60 distinct committed tokens


def test_normal_output_untouched():
    new = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert lg.check(HIST, new, 10.0) == (len(new), None)


def test_two_copies_are_allowed():
    unit = [7, 8, 9, 10]
    new = [1, 2] + unit * 2              # a speaker may repeat a phrase once
    assert lg.check(HIST, new, 10.0) == (len(new), None)


def test_three_copies_are_cut_after_the_first():
    unit = [7, 8, 9, 10, 11]
    keep, rule = lg.check(HIST, [1, 2] + unit * 4, 10.0)
    assert rule is not None and 'rep' in rule
    assert keep == 2 + len(unit)


def test_token_limit_with_two_copies():
    unit = [7, 8, 9, 10]
    new = [1, 2] + unit * 2
    keep, rule = lg.check(HIST, new, 10.0, hit_max=True)
    assert rule is not None and keep == 2 + len(unit)


def test_sentence_repeated_across_steps():
    unit = [20, 21, 22, 23, 24]
    committed = HIST + unit + unit       # two copies already committed, one per step
    keep, rule = lg.check(committed, unit, 30.0)
    assert rule == 'xrep' and keep == 0


def test_passage_translated_again():
    passage = list(range(2000, 2030))    # 30 tokens already committed
    committed = HIST + passage + [5, 6]
    keep, rule = lg.check(committed, [9] + passage, 60.0)
    assert 'lcopy' in rule and keep == 1


def test_fast_speech_is_not_cut():
    # many tokens for little audio (a long sentence said at once) is not a loop
    assert lg.check(list(range(3000, 3030)), list(range(4000, 4040)), 4.0) == (40, None)


def test_a_third_copy_needs_a_full_unit_in_the_step():
    unit = [20, 21, 22, 23]
    committed = HIST + unit + unit + unit[:3]      # third copy almost complete
    assert lg.check(committed, unit[3:], 30.0) == (1, None)
