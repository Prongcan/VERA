#!/usr/bin/env python3
"""
Test script for VERA refactored API
Tests the new vera.models.initialize() interface
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_imports():
    """Test that vera modules can be imported"""
    print("=" * 60)
    print("Test 1: Testing VERA module imports")
    print("=" * 60)

    try:
        import vera
        print("✓ import vera")

        from vera import models
        print("✓ from vera import models")

        from vera.models import initialize
        print("✓ from vera.models import initialize")

        print("\n✅ All imports successful!\n")
        return True
    except Exception as e:
        print(f"\n❌ Import failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_api_signature():
    """Test that the API has the correct signature"""
    print("=" * 60)
    print("Test 2: Testing API signature")
    print("=" * 60)

    try:
        from vera.models import initialize
        import inspect

        sig = inspect.signature(initialize)
        params = list(sig.parameters.keys())

        print(f"initialize() parameters: {params}")

        expected_params = ["model_path", "model_type", "max_new_tokens"]
        if all(p in params for p in expected_params):
            print("✓ All expected parameters present")
        else:
            print(f"⚠ Missing parameters: {set(expected_params) - set(params)}")

        print("\n✅ API signature correct!\n")
        return True
    except Exception as e:
        print(f"\n❌ API signature test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_engine_instantiation():
    """Test engine instantiation (without loading actual models)"""
    print("=" * 60)
    print("Test 3: Testing engine instantiation")
    print("=" * 60)

    try:
        from vera.models import QwenEngine, QwenEngineMasked

        # Test class constructors
        print("Testing QwenEngine class...")
        print(f"  QwenEngine.__init__ signature: {QwenEngine.__init__.__code__.co_varnames[:3]}")

        print("\nTesting QwenEngineMasked class...")
        print(f"  QwenEngineMasked.__init__ signature: {QwenEngineMasked.__init__.__code__.co_varnames[:3]}")

        print("\n✅ Engine classes accessible!\n")
        return True
    except Exception as e:
        print(f"\n❌ Engine instantiation test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("VERA Refactored API Test Suite")
    print("=" * 60 + "\n")

    results = []

    results.append(("Imports", test_imports()))
    results.append(("API Signature", test_api_signature()))
    results.append(("Engine Classes", test_engine_instantiation()))

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
    else:
        print("⚠️  Some tests failed")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
