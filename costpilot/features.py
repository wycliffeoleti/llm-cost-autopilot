import re

_INSTRUCTION_VERBS = (
    "analyze", "compare", "evaluate", "synthesize", "design",
    "critique", "contrast", "recommend", "justify", "assess",
)
_INSTRUCTION_VERB_PATTERN = re.compile(
    r"\b(?:" + "|".join(_INSTRUCTION_VERBS) + r")\b", re.IGNORECASE
)


def token_count(prompt: str) -> int:
    return len(prompt.split())


def instruction_verb_count(prompt: str) -> int:
    return len(_INSTRUCTION_VERB_PATTERN.findall(prompt))


_CONSTRAINT_PATTERNS = (
    r"\bmust\b",
    r"\bshould\b",
    r"\bdo not\b",
    r"\bdon't\b",
    r"\bat least\b",
    r"\bexactly\b",
    r"\bno more than\b",
    r"\bbetween\s+\d+\s+and\s+\d+\b",
    r"^\s*[-*]\s+\S.*$",
    r"^\s*\d+[.)]\s+\S.*$",
)
_CONSTRAINT_REGEXES = [
    re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in _CONSTRAINT_PATTERNS
]


def constraint_count(prompt: str) -> int:
    return sum(len(regex.findall(prompt)) for regex in _CONSTRAINT_REGEXES)


_CONTEXT_CUE_PATTERN = re.compile(
    r"\bthe following\b|\bgiven this\b|\bbased on this\b|\baccording to this\b",
    re.IGNORECASE,
)


def has_context(prompt: str) -> bool:
    if _CONTEXT_CUE_PATTERN.search(prompt):
        return True
    if ":" in prompt:
        trailing = prompt.split(":", 1)[1].strip()
        return len(trailing.split()) >= 5
    return False
