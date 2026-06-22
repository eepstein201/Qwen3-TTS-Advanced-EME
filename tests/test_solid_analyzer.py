"""Tests for SOLID analyzer tool."""

import pytest


class TestSOLIDScore:
    """Tests for SOLIDScore dataclass."""

    def test_solid_score_has_all_principles(self):
        """SOLIDScore has scores for all 5 principles."""
        from qwen3_tts.tools.solid_analyzer import SOLIDScore

        score = SOLIDScore(
            srp_score=8.0,
            ocp_score=7.0,
            lsp_score=9.0,
            isp_score=6.0,
            dip_score=5.0,
        )
        assert score.srp_score == 8.0
        assert score.ocp_score == 7.0
        assert score.lsp_score == 9.0
        assert score.isp_score == 6.0
        assert score.dip_score == 5.0

    def test_solid_score_calculates_total(self):
        """SOLIDScore calculates total from principle scores."""
        from qwen3_tts.tools.solid_analyzer import SOLIDScore

        score = SOLIDScore(
            srp_score=8.0,
            ocp_score=7.0,
            lsp_score=9.0,
            isp_score=6.0,
            dip_score=5.0,
        )
        assert score.total_score == 35.0

    def test_solid_score_has_violations_list(self):
        """SOLIDScore tracks violations."""
        from qwen3_tts.tools.solid_analyzer import SOLIDScore, Violation

        violations = [Violation("SRP", "Too many methods", 10)]
        score = SOLIDScore(
            srp_score=5.0,
            ocp_score=5.0,
            lsp_score=5.0,
            isp_score=5.0,
            dip_score=5.0,
            violations=violations,
        )
        assert len(score.violations) == 1
        assert score.violations[0].principle == "SRP"


class TestViolation:
    """Tests for Violation dataclass."""

    def test_violation_has_required_fields(self):
        """Violation has principle, message, and line number."""
        from qwen3_tts.tools.solid_analyzer import Violation

        v = Violation(
            principle="SRP",
            message="Class has too many public methods",
            line_number=42,
        )
        assert v.principle == "SRP"
        assert v.message == "Class has too many public methods"
        assert v.line_number == 42


class TestSRPScoring:
    """Tests for Single Responsibility Principle scoring."""

    @pytest.fixture
    def god_class_code(self):
        """Code with a class that has too many methods."""
        return '''
class GodClass:
    """A class that does too many things."""

    def method_1(self): pass
    def method_2(self): pass
    def method_3(self): pass
    def method_4(self): pass
    def method_5(self): pass
    def method_6(self): pass
    def method_7(self): pass
    def method_8(self): pass
    def method_9(self): pass
    def method_10(self): pass
    def method_11(self): pass
    def method_12(self): pass
'''

    @pytest.fixture
    def clean_class_code(self):
        """Code with a class that follows SRP."""
        return '''
class CleanClass:
    """A class with a single responsibility."""

    def do_one_thing(self):
        """Does one thing."""
        pass

    def _helper(self):
        """Private helper."""
        pass
'''

    def test_srp_score_decreases_with_many_public_methods(self, god_class_code):
        """SRP score decreases when class has too many public methods."""
        from qwen3_tts.tools.solid_analyzer import analyze_code

        result = analyze_code(god_class_code, "test_file.py")
        assert result.srp_score < 8.0  # 12 methods = lower score

    def test_srp_score_high_for_clean_class(self, clean_class_code):
        """SRP score is high for class with few public methods."""
        from qwen3_tts.tools.solid_analyzer import analyze_code

        result = analyze_code(clean_class_code, "test_file.py")
        assert result.srp_score >= 8.0  # 1 public method = good score

    def test_srp_detects_long_functions(self):
        """SRP scoring detects overly long functions."""
        long_function_code = '''
def very_long_function():
    """A function that is too long."""
    x = 1
    y = 2
    z = 3
    # ... many more lines ...
    ''' + ("\n    pass" * 80) + '''
'''
        from qwen3_tts.tools.solid_analyzer import analyze_code

        result = analyze_code(long_function_code, "test_file.py")
        assert result.srp_score < 9.0

    def test_srp_violation_includes_line_number(self, god_class_code):
        """SRP violations include line numbers."""
        from qwen3_tts.tools.solid_analyzer import analyze_code

        result = analyze_code(god_class_code, "test_file.py")
        srp_violations = [v for v in result.violations if v.principle == "SRP"]
        assert len(srp_violations) > 0
        assert all(v.line_number > 0 for v in srp_violations)


class TestOCPScoring:
    """Tests for Open/Closed Principle scoring."""

    @pytest.fixture
    def type_dispatch_code(self):
        """Code with if/else type dispatch (OCP violation)."""
        return '''
def process_data(data):
    """Process data based on type."""
    if data.type == "A":
        return process_a(data)
    elif data.type == "B":
        return process_b(data)
    elif data.type == "C":
        return process_c(data)
    elif data.type == "D":
        return process_d(data)
    elif data.type == "E":
        return process_e(data)
    else:
        return process_default(data)
'''

    @pytest.fixture
    def strategy_pattern_code(self):
        """Code using strategy pattern (OCP compliant)."""
        return '''
PROCESSORS = {
    "A": process_a,
    "B": process_b,
    "C": process_c,
}

def process_data(data):
    """Process data using strategy lookup."""
    processor = PROCESSORS.get(data.type, process_default)
    return processor(data)
'''

    def test_ocp_score_decreases_for_if_else_chains(self, type_dispatch_code):
        """OCP score decreases for if/else type dispatch."""
        from qwen3_tts.tools.solid_analyzer import analyze_code

        result = analyze_code(type_dispatch_code, "test_file.py")
        assert result.ocp_score < 10.0  # 4 elif branches = lower score

    def test_ocp_score_high_for_strategy_pattern(self, strategy_pattern_code):
        """OCP score is high for strategy pattern usage."""
        from qwen3_tts.tools.solid_analyzer import analyze_code

        result = analyze_code(strategy_pattern_code, "test_file.py")
        assert result.ocp_score >= 7.0

    def test_ocp_detects_hardcoded_mode_dispatch(self):
        """OCP detects hardcoded mode/type dispatch."""
        mode_dispatch_code = '''
def run_inference(model, text, mode):
    if mode == "clone":
        return model.generate_voice_clone(text)
    elif mode == "design":
        return model.generate_voice_design(text)
    elif mode == "custom":
        return model.generate_custom_voice(text)
    elif mode == "vllm":
        return model.generate_vllm(text)
    elif mode == "other":
        return model.generate_other(text)
'''
        from qwen3_tts.tools.solid_analyzer import analyze_code

        result = analyze_code(mode_dispatch_code, "test_file.py")
        assert result.ocp_score < 10.0  # 5 elif branches detected


class TestLSPScoring:
    """Tests for Liskov Substitution Principle scoring."""

    @pytest.fixture
    def lsp_violation_code(self):
        """Code with LSP violation (incompatible subclass method)."""
        return '''
class Base:
    def process(self, data: str) -> str:
        return data.upper()

class Derived(Base):
    def process(self, data: int) -> int:  # Incompatible signature
        return data * 2
'''

    @pytest.fixture
    def lsp_compliant_code(self):
        """Code that follows LSP."""
        return '''
class Base:
    def process(self, data: str) -> str:
        return data.upper()

class Derived(Base):
    def process(self, data: str) -> str:  # Compatible signature
        return data.lower()
'''

    def test_lsp_score_decreases_for_incompatible_signatures(self, lsp_violation_code):
        """LSP score decreases for incompatible method signatures."""
        from qwen3_tts.tools.solid_analyzer import analyze_code

        result = analyze_code(lsp_violation_code, "test_file.py")
        assert result.lsp_score < 8.0

    def test_lsp_score_high_for_compatible_signatures(self, lsp_compliant_code):
        """LSP score is high for compatible method signatures."""
        from qwen3_tts.tools.solid_analyzer import analyze_code

        result = analyze_code(lsp_compliant_code, "test_file.py")
        assert result.lsp_score >= 8.0


class TestISPScoring:
    """Tests for Interface Segregation Principle scoring."""

    @pytest.fixture
    def fat_interface_code(self):
        """Code with a fat interface (ISP violation)."""
        return '''
class MegaInterface:
    """An interface with too many methods."""

    def method_1(self): pass
    def method_2(self): pass
    def method_3(self): pass
    def method_4(self): pass
    def method_5(self): pass
    def method_6(self): pass
    def method_7(self): pass
    def method_8(self): pass
    def method_9(self): pass
    def method_10(self): pass
'''

    @pytest.fixture
    def segregated_interface_code(self):
        """Code with segregated interfaces (ISP compliant)."""
        return '''
class ReaderInterface:
    def read(self): pass

class WriterInterface:
    def write(self, data): pass

class ReaderWriter(ReaderInterface, WriterInterface):
    def read(self): return "data"
    def write(self, data): pass
'''

    def test_isp_score_decreases_for_fat_interfaces(self, fat_interface_code):
        """ISP score decreases for interfaces with many methods."""
        from qwen3_tts.tools.solid_analyzer import analyze_code

        result = analyze_code(fat_interface_code, "test_file.py")
        assert result.isp_score < 7.0

    def test_isp_score_high_for_segregated_interfaces(self, segregated_interface_code):
        """ISP score is high for small, focused interfaces."""
        from qwen3_tts.tools.solid_analyzer import analyze_code

        result = analyze_code(segregated_interface_code, "test_file.py")
        assert result.isp_score >= 7.0


class TestDIPScoring:
    """Tests for Dependency Inversion Principle scoring."""

    @pytest.fixture
    def dip_violation_code(self):
        """Code with DIP violation (direct concrete import)."""
        return '''
from concrete_module import ConcreteImplementation

class Client:
    def __init__(self):
        self.impl = ConcreteImplementation()  # Direct instantiation

    def do_work(self):
        return self.impl.process()
'''

    @pytest.fixture
    def dip_compliant_code(self):
        """Code that follows DIP (dependency injection)."""
        return '''
class Client:
    def __init__(self, implementation):  # Injected dependency
        self.impl = implementation

    def do_work(self):
        return self.impl.process()
'''

    def test_dip_score_decreases_for_direct_instantiation(self, dip_violation_code):
        """DIP score decreases for direct concrete instantiation."""
        from qwen3_tts.tools.solid_analyzer import analyze_code

        result = analyze_code(dip_violation_code, "test_file.py")
        assert result.dip_score < 10.0  # Should be lower than perfect

    def test_dip_score_high_for_dependency_injection(self, dip_compliant_code):
        """DIP score is high for dependency injection pattern."""
        from qwen3_tts.tools.solid_analyzer import analyze_code

        result = analyze_code(dip_compliant_code, "test_file.py")
        assert result.dip_score >= 8.0


class TestAnalyzeModule:
    """Tests for analyze_module function."""

    def test_analyze_module_with_real_file(self, tmp_path):
        """analyze_module works with real file path."""
        from qwen3_tts.tools.solid_analyzer import analyze_module

        # Create a simple Python file
        code_file = tmp_path / "test_module.py"
        code_file.write_text('''
class SimpleClass:
    def do_something(self):
        pass
''')

        result = analyze_module(str(code_file))
        assert result.total_score > 0
        assert isinstance(result.violations, list)

    def test_analyze_module_returns_solid_score(self, tmp_path):
        """analyze_module returns SOLIDScore instance."""
        from qwen3_tts.tools.solid_analyzer import SOLIDScore, analyze_module

        code_file = tmp_path / "test_module.py"
        code_file.write_text("def hello(): pass")

        result = analyze_module(str(code_file))
        assert isinstance(result, SOLIDScore)

    def test_analyze_module_handles_syntax_error(self, tmp_path):
        """analyze_module handles files with syntax errors."""
        from qwen3_tts.tools.solid_analyzer import analyze_module

        code_file = tmp_path / "bad_syntax.py"
        code_file.write_text("def broken(:")  # Invalid syntax

        result = analyze_module(str(code_file))
        assert result.total_score == 0  # Returns zero score on error
        assert len(result.violations) > 0


class TestAnalyzePackage:
    """Tests for analyzing an entire package."""

    def test_analyze_package_returns_dict(self, tmp_path):
        """analyze_package returns dict of module scores."""
        from qwen3_tts.tools.solid_analyzer import SOLIDScore, analyze_package

        # Create a mini package
        pkg = tmp_path / "test_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "module_a.py").write_text("def func_a(): pass")
        (pkg / "module_b.py").write_text("def func_b(): pass")

        results = analyze_package(str(pkg))
        assert isinstance(results, dict)
        assert all(isinstance(v, SOLIDScore) for v in results.values())

    def test_analyze_package_ignores_non_python_files(self, tmp_path):
        """analyze_package ignores non-Python files."""
        from qwen3_tts.tools.solid_analyzer import analyze_package

        pkg = tmp_path / "test_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "module.py").write_text("def func(): pass")
        (pkg / "readme.txt").write_text("Not Python")

        results = analyze_package(str(pkg))
        assert "module.py" in results
        assert "readme.txt" not in results


class TestCLI:
    """Tests for command-line interface."""

    def test_cli_reports_score(self, tmp_path, capsys):
        """CLI reports total SOLID score."""
        from qwen3_tts.tools.solid_analyzer import main

        code_file = tmp_path / "test.py"
        code_file.write_text("def hello(): pass")

        with pytest.MonkeyPatch.context() as m:
            m.setattr("sys.argv", ["solid_analyzer", str(code_file)])
            try:
                main()
            except SystemExit:
                pass

        captured = capsys.readouterr()
        assert "SOLID" in captured.out or "score" in captured.out.lower()

    def test_cli_fail_below_threshold(self, tmp_path):
        """CLI exits with error when score below threshold."""
        from qwen3_tts.tools.solid_analyzer import main

        # Create code with known violations - use a threshold we know it will fail
        bad_code = '''
class GodClass:
    def m1(self): pass
    def m2(self): pass
    def m3(self): pass
    def m4(self): pass
    def m5(self): pass
    def m6(self): pass
    def m7(self): pass
    def m8(self): pass
    def m9(self): pass
    def m10(self): pass
    def m11(self): pass
    def m12(self): pass
'''
        code_file = tmp_path / "bad.py"
        code_file.write_text(bad_code)

        with pytest.MonkeyPatch.context() as m:
            m.setattr("sys.argv", ["solid_analyzer", str(code_file), "--fail-below", "45"])
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1
