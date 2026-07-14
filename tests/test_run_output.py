from shorts_generator import run_output


def test_sanitize_title_replaces_spaces_with_underscores():
    assert run_output.sanitize_title("How to Build a Startup") == "How_to_Build_a_Startup"


def test_sanitize_title_strips_unsafe_characters():
    assert run_output.sanitize_title("A/B: Test?!") == "A_B_Test"


def test_sanitize_title_empty_input_falls_back_to_untitled():
    assert run_output.sanitize_title("") == "untitled"
    assert run_output.sanitize_title("???") == "untitled"


def test_sanitize_title_truncates_long_titles():
    result = run_output.sanitize_title("x" * 150)
    assert len(result) == 100
