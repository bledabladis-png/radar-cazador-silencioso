# -*- coding: utf-8 -*-
"""Test regresion: compute_pcr_signals emite skip NO_DATA cuando el
proveedor CBOE no responde. Bug preexistente desde 9a2bddcc.

Sin el fix, los return None tempranos dejaban pcr_history sin receipt y
el gate de verificacion fallaba el run entero (bug 2026-10-26).
"""
from unittest.mock import patch, MagicMock

from indicators.options import compute_pcr_signals


def test_pcr_provider_no_disponible_emite_skip_no_data():
    """Si CboeProvider.is_available()=False, skip NO_DATA."""
    with patch("indicators.options.CboeProvider") as mock_cls:
        mock_provider = MagicMock()
        mock_provider.is_available.return_value = False
        mock_cls.return_value = mock_provider

        with patch("src.pipeline.output_writer.skip") as mock_skip:
            out = compute_pcr_signals(persist_history=True)
            assert out is None
            mock_skip.assert_called_once_with('pcr_history', 'NO_DATA')


def test_pcr_provider_sin_datos_emite_skip_no_data():
    """Si get_options_data() devuelve None, skip NO_DATA."""
    with patch("indicators.options.CboeProvider") as mock_cls:
        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_provider.get_options_data.return_value = None
        mock_cls.return_value = mock_provider

        with patch("src.pipeline.output_writer.skip") as mock_skip:
            out = compute_pcr_signals(persist_history=True)
            assert out is None
            mock_skip.assert_called_once_with('pcr_history', 'NO_DATA')


def test_pcr_provider_datos_sin_total_pcr_emite_skip_no_data():
    """Si data no tiene total_pcr, skip NO_DATA."""
    with patch("indicators.options.CboeProvider") as mock_cls:
        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_provider.get_options_data.return_value = {'index_pcr': 1.0}
        mock_cls.return_value = mock_provider

        with patch("src.pipeline.output_writer.skip") as mock_skip:
            out = compute_pcr_signals(persist_history=True)
            assert out is None
            mock_skip.assert_called_once_with('pcr_history', 'NO_DATA')


def test_pcr_sin_persist_history_no_emite_skip():
    """persist_history=False: no debe emitir receipt (compatibilidad)."""
    with patch("indicators.options.CboeProvider") as mock_cls:
        mock_provider = MagicMock()
        mock_provider.is_available.return_value = False
        mock_cls.return_value = mock_provider

        with patch("src.pipeline.output_writer.skip") as mock_skip:
            out = compute_pcr_signals(persist_history=False)
            assert out is None
            mock_skip.assert_not_called()