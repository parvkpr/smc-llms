"""
Text quantification methods M: Σ* → Z.

Each method maps a string to an integer (or dict of integers) so it can be
stored as a DTMC state feature and referenced in PCTL atomic propositions.

Implemented methods (matching the LLMCHECKER paper):
  - GenderBias      : counts male/female gendered terms  → int (>0 male, <0 female)
  - SentimentScore  : polarity in [-100, 100]            → int
  - ReadingQuality  : Flesch reading ease × 100          → int
  - CopyrightSim    : Levenshtein similarity to reference string × 100 → int
  - Step            : token generation depth             → int (always available)

Quantifiers are composable: MultiQuantifier runs several at once and merges
their outputs into a single dict, which is what DTMCNode.quantification holds.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

# ── Gender word lists (matching the paper's counting approach) ────────────────
_MALE_WORDS = frozenset([
    "he", "him", "his", "himself", "man", "men", "boy", "boys",
    "male", "males", "gentleman", "gentlemen", "husband", "father",
    "son", "brother", "uncle", "nephew", "grandfather", "guy", "guys",
    "mr", "sir", "king", "prince", "actor", "waiter", "hero",
])

_FEMALE_WORDS = frozenset([
    "she", "her", "hers", "herself", "woman", "women", "girl", "girls",
    "female", "females", "lady", "ladies", "wife", "mother", "daughter",
    "sister", "aunt", "niece", "grandmother", "gal", "gals",
    "ms", "mrs", "miss", "queen", "princess", "actress", "waitress", "heroine",
])


# ── Base class ────────────────────────────────────────────────────────────────
class Quantifier(ABC):
    """Base class for all text quantification methods."""

    @abstractmethod
    def quantify(self, text: str, depth: int = 0) -> Dict[str, int]:
        """Return a dict of feature_name → integer_value."""
        ...

    def __call__(self, text: str, depth: int = 0) -> Dict[str, int]:
        return self.quantify(text, depth)


# ── Concrete quantifiers ──────────────────────────────────────────────────────
class GenderBias(Quantifier):
    """
    Gender bias score: count(male_words) - count(female_words).
    Positive → male-biased, negative → female-biased, zero → neutral.
    """

    def quantify(self, text: str, depth: int = 0) -> Dict[str, int]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        male_count = sum(1 for t in tokens if t in _MALE_WORDS)
        female_count = sum(1 for t in tokens if t in _FEMALE_WORDS)
        return {"gender": male_count - female_count}


class SentimentScore(Quantifier):
    """
    Sentiment polarity scaled to [-100, 100].
    Uses TextBlob's polarity ([-1, 1]) multiplied by 100.

    Falls back to a simple positive/negative word count if TextBlob is
    unavailable (keeps the package dependency optional).
    """

    def __init__(self) -> None:
        try:
            from textblob import TextBlob  # type: ignore
            self._textblob = TextBlob
            self._use_textblob = True
        except ImportError:
            self._use_textblob = False
            self._pos_words = frozenset([
                "good", "great", "excellent", "wonderful", "fantastic",
                "amazing", "positive", "happy", "best", "love", "nice",
            ])
            self._neg_words = frozenset([
                "bad", "terrible", "awful", "horrible", "negative",
                "sad", "worst", "hate", "poor", "difficult", "hard",
            ])

    def quantify(self, text: str, depth: int = 0) -> Dict[str, int]:
        if self._use_textblob:
            polarity = self._textblob(text).sentiment.polarity  # type: ignore
            return {"polarity": int(round(polarity * 100))}
        # Fallback
        tokens = set(re.findall(r"\b\w+\b", text.lower()))
        pos = len(tokens & self._pos_words)
        neg = len(tokens & self._neg_words)
        score = max(-100, min(100, (pos - neg) * 10))
        return {"polarity": score}


class ReadingQuality(Quantifier):
    """
    Flesch Reading Ease score × 100 → integer.
    Higher values = simpler / more accessible text.

    Uses the textstat library; falls back to a syllable-based approximation.
    """

    def __init__(self) -> None:
        try:
            import textstat  # type: ignore
            self._textstat = textstat
            self._use_textstat = True
        except ImportError:
            self._use_textstat = False

    def _approx_syllables(self, word: str) -> int:
        word = word.lower()
        count = len(re.findall(r"[aeiouy]+", word))
        if word.endswith("e") and count > 1:
            count -= 1
        return max(1, count)

    def _flesch_approx(self, text: str) -> float:
        words = re.findall(r"\b\w+\b", text)
        sentences = max(1, len(re.findall(r"[.!?]+", text)))
        if not words:
            return 100.0
        syllables = sum(self._approx_syllables(w) for w in words)
        return 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))

    def quantify(self, text: str, depth: int = 0) -> Dict[str, int]:
        if self._use_textstat:
            score = self._textstat.flesch_reading_ease(text)
        else:
            score = self._flesch_approx(text)
        return {"readability": int(round(score * 100))}


class CopyrightSim(Quantifier):
    """
    Similarity to a reference copyrighted string, as a percentage [0..100].
    Uses normalised Levenshtein edit distance so it maps cleanly to an integer.
    """

    def __init__(self, reference: str) -> None:
        self.reference = reference
        self._ref_len = len(reference)

    @staticmethod
    def _edit_distance(a: str, b: str) -> int:
        if len(a) > len(b):
            a, b = b, a
        prev = list(range(len(a) + 1))
        for j, cb in enumerate(b, 1):
            curr = [j]
            for i, ca in enumerate(a, 1):
                curr.append(min(
                    prev[i] + 1,
                    curr[i - 1] + 1,
                    prev[i - 1] + (0 if ca == cb else 1),
                ))
            prev = curr
        return prev[len(a)]

    def quantify(self, text: str, depth: int = 0) -> Dict[str, int]:
        # Compare only the last len(reference) characters to avoid penalising
        # longer generated texts that contain the reference as a substring.
        snippet = text[-self._ref_len:] if len(text) >= self._ref_len else text
        max_len = max(len(snippet), self._ref_len, 1)
        dist = self._edit_distance(snippet, self.reference)
        similarity = int(round((1.0 - dist / max_len) * 100))
        return {"similarity": similarity}


class StepCounter(Quantifier):
    """Always-available feature: current generation depth (token count)."""

    def quantify(self, text: str, depth: int = 0) -> Dict[str, int]:
        return {"step": depth}


# ── Composition ───────────────────────────────────────────────────────────────
class MultiQuantifier(Quantifier):
    """
    Runs multiple quantifiers and merges their output dicts.

    Usage:
        q = MultiQuantifier([GenderBias(), StepCounter()])
        q("The player won because he scored")
        # → {'gender': 1, 'step': 0}
    """

    def __init__(self, quantifiers: List[Quantifier]) -> None:
        self.quantifiers = quantifiers

    def quantify(self, text: str, depth: int = 0) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for q in self.quantifiers:
            result.update(q.quantify(text, depth))
        return result
