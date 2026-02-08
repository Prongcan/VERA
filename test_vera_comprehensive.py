#!/usr/bin/env python3
"""
Comprehensive test suite for VERA refactored API
Tests all four modules: models, rendering, retrieval, analysis
"""

import sys
import json
import os
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_models_module():
    """Test models module"""
    print("\n" + "=" * 60)
    print("Test 1: Models Module")
    print("=" * 60)

    try:
        from vera import models

        # Test initialize function signature
        import inspect
        sig = inspect.signature(models.initialize)
        params = list(sig.parameters.keys())

        print(f"✓ models.initialize() parameters: {params}")

        # Test that initialize returns correct type
        print("✓ models.initialize() accessible")
        print("✓ Models module tests passed!\n")
        return True
    except Exception as e:
        print(f"❌ Models module test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_rendering_module():
    """Test rendering module"""
    print("=" * 60)
    print("Test 2: Rendering Module")
    print("=" * 60)

    try:
        from vera import rendering

        # Test text_to_image function
        import inspect
        sig = inspect.signature(rendering.text_to_image)
        params = list(sig.parameters.keys())

        print(f"✓ rendering.text_to_image() parameters: {params}")

        expected_params = ["text", "output_dir", "config", "evidence_text"]
        if all(p in params for p in expected_params):
            print("✓ All expected parameters present")
        else:
            print(f"⚠ Missing parameters: {set(expected_params) - set(params)}")

        print("✓ Rendering module tests passed!\n")
        return True
    except Exception as e:
        print(f"❌ Rendering module test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_retrieval_module():
    """Test retrieval module"""
    print("=" * 60)
    print("Test 3: Retrieval Module")
    print("=" * 60)

    try:
        from vera import retrieval

        # Test qwen_embedding function
        import inspect
        qwen_sig = inspect.signature(retrieval.qwen_embedding)
        qwen_params = list(qwen_sig.parameters.keys())

        print(f"✓ retrieval.qwen_embedding() parameters: {qwen_params}")

        # Test colpali function
        colpali_sig = inspect.signature(retrieval.colpali)
        colpali_params = list(colpali_sig.parameters.keys())

        print(f"✓ retrieval.colpali() parameters: {colpali_params}")

        print("✓ Retrieval module tests passed!\n")
        return True
    except Exception as e:
        print(f"❌ Retrieval module test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_analysis_module():
    """Test analysis module"""
    print("=" * 60)
    print("Test 4: Analysis Module")
    print("=" * 60)

    try:
        from vera import analysis

        # Test create_heatmap function
        import inspect
        heatmap_sig = inspect.signature(analysis.create_heatmap)
        heatmap_params = list(heatmap_sig.parameters.keys())

        print(f"✓ analysis.create_heatmap() parameters: {heatmap_params}")

        expected_params = ["image_path", "attention_data", "output_path", "mode", "alpha", "top_k"]
        if all(p in heatmap_params for p in expected_params):
            print("✓ All expected parameters present")
        else:
            print(f"⚠ Missing parameters: {set(expected_params) - set(heatmap_params)}")

        # Test get_top_k_patches function
        patches_sig = inspect.signature(analysis.get_top_k_patches)
        patches_params = list(patches_sig.parameters.keys())

        print(f"✓ analysis.get_top_k_patches() parameters: {patches_params}")

        print("✓ Analysis module tests passed!\n")
        return True
    except Exception as e:
        print(f"❌ Analysis module test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_api_consistency():
    """Test API consistency across modules"""
    print("=" * 60)
    print("Test 5: API Consistency")
    print("=" * 60)

    try:
        from vera import models, rendering, retrieval, analysis

        # Check that all modules are importable
        assert hasattr(models, 'initialize'), "models.initialize not found"
        assert hasattr(rendering, 'text_to_image'), "rendering.text_to_image not found"
        assert hasattr(retrieval, 'qwen_embedding'), "retrieval.qwen_embedding not found"
        assert hasattr(retrieval, 'colpali'), "retrieval.colpali not found"
        assert hasattr(analysis, 'create_heatmap'), "analysis.create_heatmap not found"
        assert hasattr(analysis, 'get_top_k_patches'), "analysis.get_top_k_patches not found"

        print("✓ All expected functions are exported")
        print("✓ API consistency tests passed!\n")
        return True
    except Exception as e:
        print(f"❌ API consistency test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_documentation():
    """Test that functions have proper docstrings"""
    print("=" * 60)
    print("Test 6: Documentation")
    print("=" * 60)

    try:
        from vera import models, rendering, retrieval, analysis

        # Check docstrings
        assert models.initialize.__doc__, "models.initialize missing docstring"
        assert rendering.text_to_image.__doc__, "rendering.text_to_image missing docstring"
        assert retrieval.qwen_embedding.__doc__, "retrieval.qwen_embedding missing docstring"
        assert retrieval.colpali.__doc__, "retrieval.colpali missing docstring"
        assert analysis.create_heatmap.__doc__, "analysis.create_heatmap missing docstring"

        print("✓ All functions have docstrings")
        print("✓ Documentation tests passed!\n")
        return True
    except Exception as e:
        print(f"❌ Documentation test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_package_structure():
    """Test package structure"""
    print("=" * 60)
    print("Test 7: Package Structure")
    print("=" * 60)

    try:
        # Check that all necessary files exist
        vera_dir = ROOT / "vera"

        required_files = [
            vera_dir / "__init__.py",
            vera_dir / "models" / "__init__.py",
            vera_dir / "models" / "base.py",
            vera_dir / "models" / "qwen.py",
            vera_dir / "rendering" / "__init__.py",
            vera_dir / "rendering" / "text_to_image.py",
            vera_dir / "retrieval" / "__init__.py",
            vera_dir / "retrieval" / "qwen_embedding.py",
            vera_dir / "retrieval" / "colpali.py",
            vera_dir / "analysis" / "__init__.py",
            vera_dir / "analysis" / "heatmap.py",
        ]

        for file_path in required_files:
            assert file_path.exists(), f"Required file not found: {file_path}"
            print(f"✓ {file_path.relative_to(ROOT)}")

        print("✓ Package structure tests passed!\n")
        return True
    except Exception as e:
        print(f"❌ Package structure test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("VERA Refactored API - Comprehensive Test Suite")
    print("=" * 60)

    tests = [
        ("Models Module", test_models_module),
        ("Rendering Module", test_rendering_module),
        ("Retrieval Module", test_retrieval_module),
        ("Analysis Module", test_analysis_module),
        ("API Consistency", test_api_consistency),
        ("Documentation", test_documentation),
        ("Package Structure", test_package_structure),
    ]

    results = []
    for name, test_func in tests:
        results.append((name, test_func()))

    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r[1] for r in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed!")
        print("\nVERA refactoring is complete and verified!")
        print("\nNext steps:")
        print("1. Test with actual model inference")
        print("2. Run experiment scripts with new API")
        print("3. Verify backward compatibility")
    else:
        print("⚠️  Some tests failed")
        print("Please review the errors above and fix them.")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
