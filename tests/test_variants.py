from enunciate.recognizer import HeardPhoneme
from enunciate.variants import ENGLISH_RULES, STRICT_RULES, VariantRules, split_heard, split_rhotics


def test_split_rhotics():
    assert split_rhotics(["k", "ɑːɹ"]) == ["k", "ɑː", "ɹ"]
    assert split_rhotics(["f", "aɪɚ"]) == ["f", "aɪ", "ɚ"]
    assert split_rhotics(["ɚ", "ɹ"]) == ["ɚ", "ɹ"]


def test_split_heard_keeps_timing():
    out = split_heard([HeardPhoneme("ɔːɹ", 0.1, 0.3), HeardPhoneme("z", 0.3, 0.4)])
    assert [p.symbol for p in out] == ["ɔː", "ɹ", "z"]
    assert out[0].start == out[1].start == 0.1
    assert out[0].end == out[1].end == 0.3


def test_english_rules_accept_documented_variants():
    for expected, heard in [
        ("ɔː", "ɑː"),
        ("t", "ɾ"),
        ("d", "ɾ"),
        ("ᵻ", "ɪ"),
        ("ᵻ", "ə"),
        ("ə", "ɐ"),
    ]:
        assert ENGLISH_RULES.substitution_cost(expected, heard) == 0, (expected, heard)
        assert ENGLISH_RULES.substitution_cost(heard, expected) == 0, (heard, expected)


def test_equivalence_is_not_transitive():
    # ᵻ is a variant of both ɪ and ə, but ɪ and ə stay distinct ("ship" is not "shup").
    assert ENGLISH_RULES.substitution_cost("ɪ", "ə") == 1
    assert ENGLISH_RULES.substitution_cost("t", "d") == 1
    assert ENGLISH_RULES.substitution_cost("ɪ", "iː") == 1


def test_linking_r_rules():
    assert ENGLISH_RULES.insertion_cost("ɚ", "ɹ") == 0
    assert ENGLISH_RULES.deletion_cost("ɚ", "ɹ") == 0
    assert ENGLISH_RULES.insertion_cost("æ", "ɹ") == 1
    assert ENGLISH_RULES.insertion_cost("ɚ", "t") == 1
    assert ENGLISH_RULES.insertion_cost(None, "ɹ") == 1


def test_strict_rules_accept_nothing_but_identity():
    assert STRICT_RULES.substitution_cost("a", "a") == 0
    assert STRICT_RULES.substitution_cost("ɔː", "ɑː") == 1
    assert STRICT_RULES.insertion_cost("ɚ", "ɹ") == 1
    assert STRICT_RULES.deletion_cost("ɚ", "ɹ") == 1


def test_custom_rules():
    rules = VariantRules(
        equivalence_classes=(frozenset({"x", "y"}),), rhotic_context=frozenset({"q"})
    )
    assert rules.substitution_cost("x", "y") == 0
    assert rules.substitution_cost("x", "z") == 1
    assert rules.insertion_cost("q", "ɹ") == 0
