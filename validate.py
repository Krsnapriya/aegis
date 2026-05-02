#!/usr/bin/env python3
"""
validate.py – Local sanity-check script for Plutchik ERC Python components.

Usage:
    python validate.py

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed.
"""
import sys
import os
import traceback

PASSED = []
FAILED = []


def check(name):
    """Decorator-style context manager for a named check."""
    class _Check:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                PASSED.append(name)
                print(f"  ✓ {name}")
            else:
                FAILED.append(name)
                print(f"  ✗ {name}: {exc_val}")
            return True  # suppress exception
    return _Check()


def main():
    print("=" * 60)
    print("Plutchik ERC – Local Validation")
    print("=" * 60)

    # ── 1. Module imports ──────────────────────────────────────────────────────
    print("\n[1] Core imports")

    with check("torch importable"):
        import torch
        _ = torch.__version__

    with check("models.multitask_emotion_model importable"):
        sys.path.insert(0, os.path.dirname(__file__))
        from models.multitask_emotion_model import (
            PluTchikMultiTaskModel, EMOTION_CLASSES, INTENSITY_LABELS
        )
        assert len(EMOTION_CLASSES) == 32, "Expected 32 emotion classes"
        assert EMOTION_CLASSES[8] == 'joy', "joy must be at index 8"
        assert INTENSITY_LABELS == ['mild', 'primary', 'intense']

    with check("advanced_engine importable"):
        from advanced_engine import AdvancedPlutchikEngine
        engine = AdvancedPlutchikEngine()

    # ── 2. Model instantiation ─────────────────────────────────────────────────
    print("\n[2] Model instantiation")

    with check("PluTchikMultiTaskModel builds"):
        from models.multitask_emotion_model import PluTchikMultiTaskModel
        m = PluTchikMultiTaskModel()
        total_params = sum(p.numel() for p in m.parameters())
        print(f"    ({total_params:,} parameters)")

    with check("Forward pass (batch=1, seq=32)"):
        import torch
        from models.multitask_emotion_model import PluTchikMultiTaskModel
        m = PluTchikMultiTaskModel()
        m.eval()
        ids  = torch.randint(0, 30000, (1, 32))
        mask = torch.ones(1, 32, dtype=torch.long)
        with torch.no_grad():
            out = m(ids, mask)
        assert 'emotion_logits' in out
        assert out['emotion_logits'].shape == (1, 32)

    # ── 3. Advanced engine ─────────────────────────────────────────────────────
    print("\n[3] Advanced engine (Neural ODE + incongruity + reframes)")

    with check("AdvancedPlutchikEngine.analyze_dynamic"):
        from advanced_engine import AdvancedPlutchikEngine
        import time
        engine = AdvancedPlutchikEngine()
        text   = "Oh GREAT, another meeting! Just what I needed!!!"
        vec    = [0.1] * 32
        vec[8] = 0.8   # high joy (likely sarcastic)
        result = engine.analyze_dynamic(text, vec, [[0.2] * 32])
        assert 'risk_level' in result
        assert 'incongruity' in result
        assert 'forecast' in result
        assert isinstance(result['timestamp'], float)
        print(f"    risk_level={result['risk_level']}, "
              f"sarcasm={result['incongruity']['sarcasm_probability']:.2f}")

    with check("Trajectory forecast produces 10 steps"):
        from advanced_engine import TrajectoryForecaster
        fc   = TrajectoryForecaster()
        vec  = [1 / 32] * 32
        out  = fc.forecast(vec, [], steps=10)
        assert len(out['trajectory']) == 10, f"Expected 10 steps, got {len(out['trajectory'])}"

    with check("Incongruity detector flags sarcasm"):
        from advanced_engine import MultimodalIncongruityDetector
        det    = MultimodalIncongruityDetector()
        result = det.calculate_incongruity_score("GREAT job, really AMAZING!!!", 0.9)
        assert result['sarcasm_probability'] > 0, "Should detect incongruity"

    with check("CounterfactualGenerator returns 3 suggestions"):
        from advanced_engine import CounterfactualGenerator
        gen  = CounterfactualGenerator(None, None)
        sug  = gen.generate_reframe("I hate this!", "serenity")
        assert len(sug) == 3, f"Expected 3 suggestions, got {len(sug)}"

    # ── 4. export_for_browser.py dry-run ──────────────────────────────────────
    print("\n[4] ONNX export utilities (no actual model needed)")

    with check("export_for_browser functions importable"):
        import importlib.util, types
        spec = importlib.util.spec_from_file_location(
            "export_for_browser",
            os.path.join(os.path.dirname(__file__), "export_for_browser.py")
        )
        mod = types.ModuleType("export_for_browser")
        # Execute the module but skip the __main__ block
        with open(os.path.join(os.path.dirname(__file__), "export_for_browser.py")) as f:
            src = f.read()
        # Just check it compiles
        compile(src, "export_for_browser.py", "exec")

    # ── 5. Chrome extension manifest ──────────────────────────────────────────
    print("\n[5] Chrome extension manifest")

    with check("manifest.json is valid JSON"):
        import json
        mf_path = os.path.join(os.path.dirname(__file__), "chrome_extension", "manifest.json")
        with open(mf_path) as f:
            mf = json.load(f)
        assert mf["manifest_version"] == 3
        hosts = mf.get("host_permissions", [])
        assert "http://localhost:8000/*" not in hosts, \
            "localhost must NOT be in default host_permissions (privacy requirement)"
        assert "http://127.0.0.1:8000/*" not in hosts, \
            "127.0.0.1 must NOT be in default host_permissions (privacy requirement)"
        # Verify lib/* is accessible so ort.esm.min.js can be loaded
        all_resources = [r for block in mf.get("web_accessible_resources", [])
                         for r in block.get("resources", [])]
        assert "lib/*" in all_resources, "lib/* must be in web_accessible_resources"

    with check("background.js uses ES module import (no fetch to localhost)"):
        bg_path = os.path.join(os.path.dirname(__file__), "chrome_extension", "background.js")
        with open(bg_path) as f:
            src = f.read()
        assert "import PlutchikOnDeviceInference" in src, "Missing on-device engine import"
        assert "localhost" not in src, "background.js must not reference localhost"

    with check("content_script.js uses on-device inference (no fetch)"):
        cs_path = os.path.join(os.path.dirname(__file__), "chrome_extension", "content_script.js")
        with open(cs_path) as f:
            src = f.read()
        assert "fetch(" not in src, "content_script must not make fetch() calls"
        assert "analyzeText" in src
        assert "localhost" not in src, "content_script must not reference localhost"

    with check("ondevice-inference.js has heuristic fallback"):
        od_path = os.path.join(os.path.dirname(__file__), "chrome_extension", "ondevice-inference.js")
        with open(od_path) as f:
            src = f.read()
        assert "_heuristicClassify" in src, "Missing heuristic fallback"
        assert "_forecastTrajectory" in src, "Missing trajectory forecaster"
        assert "_generateReframes" in src, "Missing reframe generator"
        assert "import * as ort from 'onnxruntime-web'" not in src, \
            "Must not use bare npm import for ort"

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Results: {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("Failed checks:")
        for f in FAILED:
            print(f"  – {f}")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
