from costpilot.verification import simulated_agreement_score


def test_simulated_agreement_ignores_fake_model_identity():
    original = "[claude-haiku] simulated response (digest=abcd1234, input_tokens=5)"
    reference = "[gpt-4o] simulated response (digest=abcd1234, input_tokens=5)"
    assert simulated_agreement_score(original, reference) == 1.0


def test_simulated_agreement_returns_zero_for_different_fake_payloads():
    original = "[claude-haiku] simulated response (digest=abcd1234, input_tokens=5)"
    reference = "[gpt-4o] simulated response (digest=deadbeef, input_tokens=5)"
    assert simulated_agreement_score(original, reference) == 0.0
