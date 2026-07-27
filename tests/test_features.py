from costpilot.features import token_count, instruction_verb_count
from costpilot.features import constraint_count


def test_token_count_counts_whitespace_separated_words():
    assert token_count("one two three") == 3


def test_token_count_handles_single_word():
    assert token_count("hello") == 1


def test_instruction_verb_count_counts_known_verbs():
    assert instruction_verb_count("Analyze and compare these two options.") == 2


def test_instruction_verb_count_is_case_insensitive():
    assert instruction_verb_count("ANALYZE this data.") == 1


def test_instruction_verb_count_returns_zero_for_plain_question():
    assert instruction_verb_count("What is the capital of France?") == 0


def test_constraint_count_counts_constraint_keywords():
    assert constraint_count("You must include at least three examples.") == 2


def test_constraint_count_counts_bulleted_lines():
    assert constraint_count("- one\n- two\n- three") == 3


def test_constraint_count_returns_zero_for_plain_question():
    assert constraint_count("What is the capital of France?") == 0
