import unittest


class TestClaudeMD(unittest.TestCase):
    def test_claude_md_under_300_lines(self):
        with open("CLAUDE.md") as f:
            lines = f.readlines()
        self.assertLessEqual(len(lines), 300,
            f"CLAUDE.md is {len(lines)} lines; must be ≤300 for progressive disclosure")


if __name__ == "__main__":
    unittest.main()
