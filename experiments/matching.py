"""Boundary-aware text matching, shared by the label validator and the experiment metrics.

Plain substring search gives wrong answers on policy text: "4 hours" is inside "24 hours" and "500" is
inside "1,500". contains_phrase() refuses those matches.
"""
import re


def norm(text: str) -> str:
    """Lowercase, unify quotes/dashes/rupee spacing, collapse whitespace."""
    for fancy, plain in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
                         ("–", "-"), ("—", "-"), (" ", " "), ("₹ ", "₹")):
        text = text.replace(fancy, plain)
    return " ".join(text.lower().split())


def canon(text: str) -> str:
    """norm() plus tidy table pipes, so a pasted table row compares equal to the document's row."""
    return re.sub(r"\s*\|\s*", "|", norm(text)).strip("|")


def contains_phrase(haystack: str, phrase: str) -> bool:
    """Boundary-aware containment on norm()-ed text. The phrase must not be glued to other letters or
    digits, and a number must not be the tail of a bigger one ("4 hours" in "24 hours", "500" in "1,500")."""
    p = re.escape(norm(phrase))
    pattern = r"(?<![a-z0-9])(?<![0-9][,.])" + p + r"(?![a-z0-9])(?![,.][0-9])"
    return re.search(pattern, haystack) is not None
