import random

import pytest

from enunciate.align import AlignedPair, align, total_penalty


def ops(path: list[AlignedPair]) -> list[str]:
    return [pair.op for pair in path]


def reference_distance(a: list[str], b: list[str]) -> int:
    """Textbook Levenshtein distance, used to cross-check the alignment cost."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def test_identical_sequences_match_everywhere():
    path = align(["a", "b", "c"], ["a", "b", "c"])
    assert ops(path) == ["match", "match", "match"]
    assert total_penalty(path) == 0
    assert [p.expected_index for p in path] == [0, 1, 2]
    assert [p.heard_index for p in path] == [0, 1, 2]


def test_substitution_deletion_insertion():
    assert ops(align(["a", "b"], ["a", "x"])) == ["match", "substitute"]
    assert ops(align(["a", "b"], ["a"])) == ["match", "delete"]
    assert ops(align(["a"], ["a", "b"])) == ["match", "insert"]


def test_empty_sequences():
    assert ops(align([], ["a", "b"])) == ["insert", "insert"]
    assert ops(align(["a", "b"], [])) == ["delete", "delete"]
    assert align([], []) == []


def test_insertion_is_attached_after_the_phoneme_it_follows():
    path = align(["a", "b"], ["a", "x", "b"])
    assert ops(path) == ["match", "insert", "match"]
    assert path[1].heard_index == 1


def test_substitution_preferred_over_insert_plus_delete():
    path = align(["a"], ["b"])
    assert ops(path) == ["substitute"]
    assert total_penalty(path) == 1


@pytest.mark.parametrize("seed", range(25))
def test_cost_equals_levenshtein_distance_and_path_is_consistent(seed: int):
    rng = random.Random(seed)
    alphabet = ["a", "b", "c", "d"]
    expected = [rng.choice(alphabet) for _ in range(rng.randint(0, 12))]
    heard = [rng.choice(alphabet) for _ in range(rng.randint(0, 12))]
    path = align(expected, heard)
    assert total_penalty(path) == reference_distance(expected, heard)
    # Every index appears exactly once and in order on each side.
    assert [p.expected_index for p in path if p.expected_index is not None] == list(
        range(len(expected))
    )
    assert [p.heard_index for p in path if p.heard_index is not None] == list(range(len(heard)))
    for pair in path:
        if pair.op == "match":
            assert expected[pair.expected_index] == heard[pair.heard_index]
        if pair.op == "substitute":
            assert expected[pair.expected_index] != heard[pair.heard_index]


def test_custom_substitution_cost_marks_accepted_variant():
    def lenient(expected: str, heard: str) -> int:
        return 0 if {expected, heard} <= {"t", "ɾ"} else int(expected != heard)

    path = align(["w", "t", "ɚ"], ["w", "ɾ", "ɚ"], substitution=lenient)
    assert ops(path) == ["match", "substitute", "match"]
    assert path[1].penalty == 0
    assert total_penalty(path) == 0


def test_context_dependent_gap_costs():
    def free_linking_r(previous: str | None, symbol: str) -> int:
        return 0 if symbol == "ɹ" and previous == "ɚ" else 1

    path = align(["ɚ", "ɪ", "z"], ["ɚ", "ɹ", "ɪ", "z"], insertion=free_linking_r)
    assert ops(path) == ["match", "insert", "match", "match"]
    assert path[1].penalty == 0
    # The same ɹ in a different context still costs.
    path = align(["b", "ɪ", "z"], ["b", "ɹ", "ɪ", "z"], insertion=free_linking_r)
    assert total_penalty(path) == 1
    # And a free deletion.
    path = align(["ɚ", "ɹ", "ə"], ["ɚ", "ə"], deletion=free_linking_r)
    assert ops(path) == ["match", "delete", "match"]
    assert total_penalty(path) == 0
