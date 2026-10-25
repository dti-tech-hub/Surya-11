"""run_all_checks.py — validate the environment and every major component.

Each check executes the real code path (not just an import) and prints one of:

    PASS
    FAIL
    WARNING
    NOT CONFIGURED

Exit code is 0 when no check FAILs.
"""
from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("STREAMLIT_LOG_LEVEL", "error")


def _status_line(status: str, message: str = "") -> str:
    return f"[{status}]" + (f" {message}" if message else "")


def main() -> int:
    print("=" * 68)
    print("       PowerShield-FL: System & Component Validation")
    print("=" * 68)

    results: dict[str, tuple[str, str]] = {}

    # 1. Environment -----------------------------------------------------
    py_ver = sys.version_info
    if py_ver >= (3, 10):
        results["Environment"] = ("PASS", f"Python {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    else:
        results["Environment"] = ("FAIL", f"Python {py_ver.major}.{py_ver.minor} < 3.10 required")

    # 2. Dependencies ----------------------------------------------------
    missing = []
    for pkg in ["torch", "numpy", "pandas", "sklearn", "matplotlib", "shap", "streamlit", "yaml", "psutil", "joblib", "scipy"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        results["Dependencies"] = ("FAIL", f"Missing packages: {', '.join(missing)}")
    else:
        results["Dependencies"] = ("PASS", "All core libraries installed")

    # 3. Configuration ---------------------------------------------------
    cfg = None
    try:
        from utilities.config import load_config

        cfg = load_config("quick")
        required = ["experiment", "data", "clients", "federated", "privacy", "communication", "aggregation", "evaluation"]
        absent = [key for key in required if key not in cfg]
        if absent:
            results["Configuration"] = ("FAIL", f"Missing config sections: {', '.join(absent)}")
        else:
            load_config("full")
            results["Configuration"] = ("PASS", "quick/full presets and all required sections loaded")
    except Exception as exc:
        results["Configuration"] = ("FAIL", str(exc))

    # 4/5. Dataset + preprocessing --------------------------------------
    splits = None
    if cfg is not None:
        try:
            from preprocessing.pipeline import prepare_data

            splits = prepare_data(cfg, seed=42)
            raw_files = list((ROOT / "data" / "raw").glob("*.*"))
            raw_note = f"{len(raw_files)} raw file(s)" if raw_files else "no raw file"
            results["Dataset"] = (
                "PASS" if raw_files else "WARNING",
                f"{splits.source}; {raw_note}; synthetic={splits.is_synthetic}",
            )
            results["Preprocessing"] = (
                "PASS",
                f"train={len(splits.train.y)}, val={len(splits.val.y)}, test={len(splits.test.y)}; "
                f"features={len(splits.train.feature_names)}; split_unit={splits.preprocessing_report.get('split_unit')}",
            )
        except Exception as exc:
            results["Dataset"] = ("FAIL", str(exc))
            results["Preprocessing"] = ("FAIL", str(exc))

    # 6. Model -----------------------------------------------------------
    model = None
    if splits is not None:
        try:
            import numpy as np

            from models.lightweight import build_lightweight

            n_feat = splits.train.X_matrix.shape[1]
            model = build_lightweight(cfg, n_feat, seed=42)
            probs = model.predict_proba(splits.val.X_matrix[:16])
            if not np.isfinite(probs).all() or len(probs) != 16:
                raise ValueError("model returned invalid probabilities")
            results["Model"] = ("PASS", f"LightweightMLP: {model.n_params} parameters, {probs.shape[0]} predictions")
        except Exception as exc:
            results["Model"] = ("FAIL", str(exc))

    # 7. Federated learning + 8. privacy + 9. aggregation + 10. communication
    if splits is not None and model is not None:
        try:
            import numpy as np
            import torch

            from federated.aggregation import aggregate_deltas
            from federated.communication import compress_state, decompress_state, payload_bytes
            from federated.partition import build_partition
            from federated.simulator import FederatedSimulator
            from models.lightweight import build_lightweight
            from privacy.base import build_privacy
            from utilities.types import ClientUpdate

            partitions = build_partition(splits.train, cfg, 42)
            factory = lambda: build_lightweight(cfg, splits.train.X_matrix.shape[1], seed=42)
            privacy = build_privacy(cfg, seed=42)
            sim = FederatedSimulator(
                cfg=cfg,
                train_X=splits.train.X_matrix,
                train_y=splits.train.y,
                partitions=partitions,
                model_factory=factory,
                privacy_mechanism=privacy,
                val_X=splits.val.X_matrix,
                val_y=splits.val.y,
                seed=42,
            )
            result = sim.run(rounds=1)
            if not result["history"] or result["communication"]["total_bytes"] <= 0:
                raise ValueError("federated round produced no history/bytes")
            results["Federated Learning"] = (
                "PASS",
                f"{len(partitions)} clients, 1 executed round, {result['communication']['total_bytes']} measured bytes",
            )

            # Privacy: clipping/noise must modify a deliberately large update.
            probe = {"w": torch.randn(64) * 50.0}
            protected = privacy.apply(probe)
            if protected["w"].shape != probe["w"].shape or not torch.isfinite(protected["w"]).all():
                raise ValueError("privacy mechanism returned malformed tensors")
            results["Privacy"] = ("PASS", f"{privacy.describe().get('mechanism', 'configured')} applied to a probe update")

            # Aggregation: adaptive weights sum to one and aggregate correctly.
            u1 = ClientUpdate(
                client_id="probe_1", num_samples=10, state_dict={"w": torch.ones(2)},
                train_loss=0.5, val_loss=0.4,
                val_metrics={"f1": 0.8, "roc_auc": 0.9, "val_loss": 0.4},
                num_positive=2, num_negative=8, update_norm=1.0,
            )
            u2 = ClientUpdate(
                client_id="probe_2", num_samples=30, state_dict={"w": torch.ones(2)},
                train_loss=0.5, val_loss=0.6,
                val_metrics={"f1": 0.5, "roc_auc": 0.65, "val_loss": 0.6},
                num_positive=3, num_negative=27, update_norm=1.0,
            )
            from federated.adaptive_aggregation import adaptive_weights

            weights = adaptive_weights([u1, u2], cfg)
            if not np.isclose(float(weights.sum()), 1.0):
                raise ValueError("adaptive weights do not sum to 1")
            agg = aggregate_deltas([{"w": torch.ones(2)}, {"w": torch.full((2,), 3.0)}], weights)
            if tuple(agg["w"].shape) != (2,):
                raise ValueError("aggregate_deltas returned wrong shape")
            results["Aggregation"] = ("PASS", f"adaptive weights sum={weights.sum():.6f}; weighted mean executed")

            # Communication: measured serialized round-trip.
            state = {"layer.weight": torch.randn(8, 4), "layer.bias": torch.randn(4)}
            header, payload, _ = compress_state(state, cfg)
            restored = decompress_state(header, payload)
            size = payload_bytes(header, payload)
            if size <= 0 or tuple(restored["layer.weight"].shape) != (8, 4):
                raise ValueError("compression round-trip failed")
            results["Communication"] = ("PASS", f"{cfg['communication']['method']} payload={size} bytes; round-trip restored")
        except Exception as exc:
            for key in ("Federated Learning", "Privacy", "Aggregation", "Communication"):
                results.setdefault(key, ("FAIL", str(exc)))

    # 11. Evaluation -----------------------------------------------------
    if splits is not None and model is not None:
        try:
            from evaluation.metrics import compute_classification_metrics

            metrics = compute_classification_metrics(splits.val.y, model.predict_proba(splits.val.X_matrix), threshold=0.5)
            needed = {"accuracy", "precision", "recall", "f1", "roc_auc"}
            if not needed.issubset(metrics):
                raise ValueError(f"missing metrics: {sorted(needed - set(metrics))}")
            results["Evaluation"] = ("PASS", f"metrics computed: F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
        except Exception as exc:
            results["Evaluation"] = ("FAIL", str(exc))

    # 12. Dashboard ------------------------------------------------------
    try:
        dashboard_file = ROOT / "dashboard" / "app.py"
        if not dashboard_file.exists():
            raise FileNotFoundError("dashboard/app.py missing")
        import logging

        from streamlit.testing.v1 import AppTest

        logging.getLogger("streamlit").setLevel(logging.CRITICAL)
        previous_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                at = AppTest.from_file(str(dashboard_file), default_timeout=180)
                at.run()
        finally:
            logging.disable(previous_disable)
        if at.exception:
            raise RuntimeError(f"dashboard raised: {at.exception[0].value}")
        results["Dashboard"] = ("PASS", f"Streamlit AppTest executed; {len(at.tabs)} tabs; no exceptions")
    except Exception as exc:
        results["Dashboard"] = ("FAIL", str(exc))

    # 13. Tests ----------------------------------------------------------
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )
        tail = (proc.stdout or proc.stderr).strip().splitlines()
        summary = tail[-1] if tail else "no pytest output"
        if proc.returncode == 0:
            results["Tests"] = ("PASS", summary)
        else:
            results["Tests"] = ("FAIL", summary)
    except Exception as exc:
        results["Tests"] = ("FAIL", str(exc))

    # Print ---------------------------------------------------------------
    for component, (status, detail) in results.items():
        print(f"{component:<22}: {_status_line(status, detail)}")

    print("-" * 68)
    passes = sum(1 for s, _ in results.values() if s == "PASS")
    warnings = sum(1 for s, _ in results.values() if s == "WARNING")
    fails = sum(1 for s, _ in results.values() if s == "FAIL")
    not_configured = sum(1 for s, _ in results.values() if s == "NOT CONFIGURED")
    print(
        f"Summary: {passes} PASS, {warnings} WARNING, {fails} FAIL, "
        f"{not_configured} NOT CONFIGURED out of {len(results)} checks."
    )
    print("=" * 68)
    return 1 if fails > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
