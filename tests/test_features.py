from costpilot.features import token_count


def test_token_count_counts_whitespace_separated_words():
    assert token_count("one two three") == 3


def test_token_count_handles_single_word():
    assert token_count("hello") == 1
