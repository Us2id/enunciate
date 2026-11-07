"""Edit-distance alignment between expected and heard phoneme sequences."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

Op = Literal["match", "substitute", "delete", "insert"]

SubstitutionCost = Callable[[str, str], int]
GapCost = Callable[[str | None, str], int]


@dataclass(frozen=True)
class AlignedPair:
    """One step of an alignment path.

    ``expected_index`` is ``None`` for insertions (a heard phoneme with no
    counterpart) and ``heard_index`` is ``None`` for deletions. ``penalty`` is
    the step's cost: zero for exact matches and for differences the cost
    functions accept as variants.
    """

    op: Op
    expected_index: int | None
    heard_index: int | None
    penalty: int = 1


def _default_substitution(expected: str, heard: str) -> int:
    return 0 if expected == heard else 1


def _default_gap(_previous_expected: str | None, _symbol: str) -> int:
    return 1


def align(
    expected: Sequence[str],
    heard: Sequence[str],
    *,
    substitution: SubstitutionCost = _default_substitution,
    insertion: GapCost = _default_gap,
    deletion: GapCost = _default_gap,
) -> list[AlignedPair]:
    """Minimum-cost alignment with a deterministic backtrace.

    ``substitution(expected, heard)`` prices aligning two symbols, and
    ``insertion``/``deletion`` price a gap given the expected symbol just
    before it, so context-dependent rules (a linking ɹ after an r-coloured
    vowel) can be free. Ties are broken in favour of matches and substitutions,
    then deletions, then insertions, so an extra sound is attached after the
    phoneme it follows.
    """
    n, m = len(expected), len(heard)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = cost[i - 1][0] + deletion(_before(expected, i - 1), expected[i - 1])
    for j in range(1, m + 1):
        cost[0][j] = cost[0][j - 1] + insertion(None, heard[j - 1])
    for i in range(1, n + 1):
        previous, current = cost[i - 1], cost[i]
        symbol = expected[i - 1]
        delete_cost = deletion(_before(expected, i - 1), symbol)
        for j in range(1, m + 1):
            current[j] = min(
                previous[j - 1] + substitution(symbol, heard[j - 1]),
                previous[j] + delete_cost,
                current[j - 1] + insertion(symbol, heard[j - 1]),
            )

    path: list[AlignedPair] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            step = substitution(expected[i - 1], heard[j - 1])
            if cost[i][j] == cost[i - 1][j - 1] + step:
                op: Op = "match" if expected[i - 1] == heard[j - 1] else "substitute"
                path.append(AlignedPair(op, i - 1, j - 1, step))
                i -= 1
                j -= 1
                continue
        if i > 0:
            step = deletion(_before(expected, i - 1), expected[i - 1])
            if cost[i][j] == cost[i - 1][j] + step:
                path.append(AlignedPair("delete", i - 1, None, step))
                i -= 1
                continue
        step = insertion(_before(expected, i), heard[j - 1])
        path.append(AlignedPair("insert", None, j - 1, step))
        j -= 1
    path.reverse()
    return path


def _before(expected: Sequence[str], index: int) -> str | None:
    """The expected symbol preceding position ``index``, if any."""
    return expected[index - 1] if index > 0 else None


def total_penalty(path: Sequence[AlignedPair]) -> int:
    return sum(pair.penalty for pair in path)
