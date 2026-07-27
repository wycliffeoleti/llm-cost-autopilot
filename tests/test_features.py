from costpilot.features import token_count, instruction_verb_count
from costpilot.features import constraint_count
from costpilot.features import has_context
from costpilot.features import output_format_complexity


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


def test_has_context_true_for_colon_delimited_data_block():
    prompt = "Extract the total from this invoice: Subtotal $10, Tax $1, Total $11."
    assert has_context(prompt) is True


def test_has_context_true_for_the_following_cue():
    prompt = "Summarize the following report and highlight the key risks it describes in detail."
    assert has_context(prompt) is True


def test_has_context_false_for_plain_question():
    assert has_context("What is the capital of France?") is False


def test_output_format_complexity_strict_for_json_request():
    assert output_format_complexity("Return the result as JSON with fields name and age.") == 2


def test_output_format_complexity_simple_for_list_request():
    assert output_format_complexity("Reformat this into a bulleted list.") == 1


def test_output_format_complexity_zero_for_free_text():
    assert output_format_complexity("What is the capital of France?") == 0
