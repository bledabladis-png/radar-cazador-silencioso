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
            time.sleep(backoff * attempt)
    raise last_exc
