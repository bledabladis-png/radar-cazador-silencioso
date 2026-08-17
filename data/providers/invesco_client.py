# -*- coding: utf-8 -*-
"""
Cliente reutilizable para la API pública de Invesco.

Fuente:
    https://dng-api.invesco.com

Uso:
    from data.providers.invesco_client import InvescoClient

    api = InvescoClient(cusip="46090E103")
    dataset = api.fetch_all()

Este módulo SOLO obtiene datos crudos.
No calcula flujos, señales ni métricas de Radar.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from curl_cffi import requests as curl_requests


logger = logging.getLogger("invesco_client")


class InvescoClientError(Exception):
    pass


class InvescoClient:
    """Cliente genérico Invesco por CUSIP."""

    BASE_URL = "https://dng-api.invesco.com/cache/v1/accounts"

    HEADERS = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "en-US,en;q=0.9",
        "cache-control": "no-cache",
        "origin": "https://www.invesco.com",
        "pragma": "no-cache",
        "referer": "https://www.invesco.com/",
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/151.0.0.0 Safari/537.36"
        ),
    }

    def __init__(
        self,
        cusip: str,
        locale: str = "en_US",
        delay: float = 0.25,
        retries: int = 3,
        session: Any | None = None,
    ) -> None:
        self.cusip = cusip
        self.locale = locale
        self.delay = delay
        self.retries = retries

        self.base = (
            f"{self.BASE_URL}/{locale}/shareclasses/{cusip}"
        )

        if session is None:
            self.session = curl_requests.Session(impersonate="chrome")
        else:
            self.session = session

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------
    def _request(
        self,
        path: str,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        path = "/" + path.lstrip("/")
        params = params or {}

        cache_buster = str(int(time.time() * 1000))
        query = [f"{key}={value}" for key, value in params.items()]
        query.append(f"_cb={cache_buster}")

        url = f"{self.base}{path}?{'&'.join(query)}"

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self.session.get(
                    url,
                    headers=self.HEADERS,
                    timeout=30,
                )
                response.raise_for_status()
                body = response.text.strip()
                if not body:
                    raise InvescoClientError("Respuesta vacía")
                try:
                    return response.json()
                except json.JSONDecodeError as exc:
                    raise InvescoClientError(f"JSON inválido: {exc}") from exc

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Intento %s/%s fallido en %s: %s",
                    attempt,
                    self.retries,
                    path,
                    exc,
                )
                if attempt < self.retries:
                    time.sleep(self.delay * attempt)

        raise InvescoClientError(
            f"Fallo definitivo en {path}: {last_error}"
        )

    # ------------------------------------------------------------------
    # ENDPOINTS
    # ------------------------------------------------------------------
    def navs(self) -> dict[str, Any]:
        """NAV histórico oficial."""
        return self._request(
            "/navs",
            {
                "idType": "cusip",
                "productType": "ETF",
            },
        )

    def prices(self) -> dict[str, Any]:
        """Snapshot actual con sharesOutstanding."""
        return self._request(
            "/prices",
            {
                "idType": "cusip",
                "variationType": "priceListing",
                "productType": "ETF",
                "productSubType": "ETF",
            },
        )

    def key_stats(self) -> dict[str, Any]:
        return self._request(
            "/keyStats",
            {
                "idType": "cusip",
                "productType": "ETF",
            },
        )

    def performance(self) -> dict[str, Any]:
        return self._request(
            "/performance/standard",
            {
                "idType": "cusip",
                "productType": "ETF",
                "performanceSubType": "annualized",
                "performancePeriod": "monthly",
            },
        )

    def holdings(self) -> dict[str, Any]:
        return self._request(
            "/holdings/fund",
            {
                "idType": "cusip",
                "productType": "ETF",
                "loadType": "initial",
            },
        )

    def sectors(self) -> dict[str, Any]:
        return self._request(
            "/weightedHoldings/fund",
            {
                "idType": "cusip",
                "productType": "ETF",
                "breakdown": "sector",
            },
        )

    def distributions(self) -> dict[str, Any]:
        return self._request(
            "/distribution",
            {
                "idType": "cusip",
                "productType": "ETF",
                "loadType": "initial",
            },
        )

    def frequency_premium_discount(self) -> dict[str, Any]:
        return self._request(
            "/frequencyPremiumDiscounts",
            {
                "idType": "cusip",
                "productType": "ETF",
                "loadType": "initial",
            },
        )

    def premium_discount(self) -> dict[str, Any]:
        return self._request(
            "/navs",
            {
                "idType": "cusip",
                "productType": "ETF",
                "variationType": "premiumDiscounts",
            },
        )

    # ------------------------------------------------------------------
    # DATASET COMPLETO
    # ------------------------------------------------------------------
    def fetch_all(self) -> dict[str, Any]:
        logger.info("Dataset completo | CUSIP=%s", self.cusip)

        dataset: dict[str, Any] = {
            "metadata": {
                "cusip": self.cusip,
                "source": "Invesco",
                "api_base": self.base,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }
        }

        endpoints = {
            "navs": self.navs,
            "prices": self.prices,
            "key_stats": self.key_stats,
            "performance": self.performance,
            "holdings": self.holdings,
            "sectors": self.sectors,
            "distributions": self.distributions,
            "frequency_premium_discount": self.frequency_premium_discount,
            "premium_discount": self.premium_discount,
        }

        for name, function in endpoints.items():
            try:
                dataset[name] = function()
                logger.info("OK | %s", name)
            except Exception as exc:
                logger.error("ERROR | %s | %s", name, exc)
                dataset[name] = {"_error": str(exc)}
            time.sleep(self.delay)

        return dataset
