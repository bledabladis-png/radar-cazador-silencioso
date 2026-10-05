# -*- coding: utf-8 -*-
"""Tests orquestador Capa 1.

No ejecuta sobre datos reales. Solo verifica CLI y dry-run.

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gold_standard import run_capa1


def test_parse_args_default_dry_run():
    args = run_capa1.parse_args([])
    assert args.go is False
    assert args.skip_power is False
    assert args.n_B is None


def test_parse_args_go():
    args = run_capa1.parse_args(["--go"])
    assert args.go is True


def test_parse_args_n_B():
    args = run_capa1.parse_args(["--go", "--n-B", "400"])
    assert args.n_B == 400


def test_parse_args_skip_power():
    args = run_capa1.parse_args(["--skip-power"])
    assert args.skip_power is True


def test_sha256_file(tmp_path):
    p = tmp_path / "x.txt"
    p.write_bytes(b"hello")
    # sha256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
    expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert run_capa1.sha256_file(p) == expected


def test_dry_run_no_toca_disco(tmp_path, capsys):
    """Dry-run con --out-dir inexistente no debe crearlo."""
    out = tmp_path / "no_debe_existir"
    rc = run_capa1.main(["--out-dir", str(out)])
    assert rc == 0
    assert not out.exists()


def test_dry_run_imprime_dry_run(tmp_path, capsys):
    out = tmp_path / "x"
    run_capa1.main(["--out-dir", str(out)])
    captured = capsys.readouterr()
    assert "DRY-RUN" in captured.out