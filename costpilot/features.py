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
