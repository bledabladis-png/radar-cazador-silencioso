"""DT3 Fase 0: generador del golden de caracterizacion.

Ejecutar una vez: py tests/fixtures/_gen_darkpool_golden.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

from tests.fixtures._darkpool_setup import setup_and_run


class _FakeMonkey:
    def __init__(self):
        self.originals = []

    def chdir(self, p):
        import os
        self.originals.append(("cwd", os.getcwd()))
        os.chdir(p)

    def setattr(self, obj, name, val):
        self.originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, val)

    def undo(self):
        import os
        for item in reversed(self.originals):
            if item[0] == "cwd":
                os.chdir(item[1])
            else:
                obj, name, val = item
                setattr(obj, name, val)


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        m = _FakeMonkey()
        try:
            res = setup_and_run(m, tmp)
        finally:
            m.undo()

    if res is None:
        raise SystemExit("compute_darkpool_signals devolvio None")

    def _clean(v):
        if hasattr(v, "item"):
            return v.item()
        return v

    golden = {
        "reference_date": "2026-09-17",
        "generator": "tests/fixtures/_gen_darkpool_golden.py",
        "setup": "tests/fixtures/_darkpool_setup.py",
        "seed": 42,
        "latest_week": res["week"],
        "contract": {
            "media_dark_pool": _clean(res["media_dark_pool"]),
            "n_tickers_ats": _clean(res["n_tickers_ats"]),
            "n_tickers_total": _clean(res["n_tickers_total"]),
            "z_score": _clean(res["z_score"]),
            "week": res["week"],
            "status": res["status"],
        },
        "characterization": {
            "state": res["state"],
            "momentum": _clean(res["momentum"]),
            "percentile": _clean(res["percentile"]),
            "fecha_pre_fase1": res["fecha"],
            "z_windows": {
                k: ({"z": _clean(v["z"]), "state": v["state"]} if v else None)
                for k, v in res["z_windows"].items()
            },
        },
    }
    out = Path("tests/fixtures/darkpool_golden.json")
    out.write_bytes(json.dumps(golden, indent=2, ensure_ascii=False).encode("utf-8") + b"\n")
    print(f"OK -> {out}")
    print(f"status={res['status']} week={res['week']} fecha={res['fecha']}")
    print(f"media_dark_pool={res['media_dark_pool']}")
    print(f"z_score={res['z_score']} state={res['state']}")
    print(f"n_ats={res['n_tickers_ats']} n_total={res['n_tickers_total']}")


if __name__ == "__main__":
    main()