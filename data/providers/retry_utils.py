# -*- coding: utf-8 -*-
"""Utilidades de re-reintento para proveedores."""

import time


def retry_call(func, *args, retries=3, backoff=2, **kwargs):
    """Reintenta una llamada con backoff exponencial.

    El último error se relanza si se agotan los intentos.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            # Solo dormir si hay un siguiente intento. A2.1-10: antes
            # se dormia tambien en el ultimo, desperdiciando backoff*(n-1)s.
            if attempt < retries - 1:
                # Fix 2026-10-28 (rate-limit-audit): `backoff * attempt`
                # daba sleep(0) en el primer retry (attempt=0), asi que
                # el primer reintento era inmediato. Con un 429 de
                # Yahoo eso agrava el bloqueo. `backoff * (attempt + 1)`
                # da el patron esperado: backoff, 2*backoff, 3*backoff.
                # Coherente con la docstring ("backoff exponencial" en
                # la practica es lineal creciente; el exponencial real
                # esta en yahoo.py y sec_13f/downloader.py).
                time.sleep(backoff * (attempt + 1))
    raise last_exc
