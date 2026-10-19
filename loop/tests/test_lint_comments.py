import subprocess

import lint_comments

BEFORE = """def f(x):
    # an old comment that is long enough to count as eight words
    return x
"""

AFTER = '''def f(x):
    # an old comment that is long enough to count as eight words
    return x  # short


def g(x):
    """Return the input unchanged after doing nothing at all to it."""
    # This now uses the optimized version of the fused kernel here.
    source = """
    float y = x; // the kernel keeps FP32 here because of the rounding
    /* this block comment also runs well past the eight word limit */
    """
    return x
'''


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_flags_only_added_long_texts(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@example.com")
    git(tmp_path, "config", "user.name", "t")
    (tmp_path / "m.py").write_text(BEFORE)
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "a")
    base = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout
    (tmp_path / "m.py").write_text(AFTER)
    git(tmp_path, "commit", "-qam", "b")
    flagged = lint_comments.flag(tmp_path, base.strip())
    text = "\n".join(flagged)
    assert len(flagged) == 4, flagged
    assert "Return the input" in text
    assert "optimized version" in text
    assert "rounding" in text
    assert "an old comment" not in text
