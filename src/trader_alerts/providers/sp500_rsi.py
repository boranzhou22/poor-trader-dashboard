from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any

import requests

from ..constants import IndicatorId
from ..models import Observation
from .base import Provider


class Sp500RsiProvider(Provider):
    """
    S&P 500 RSI 数据源（按“可解析到 RSI(14) Value”的顺序尝试）：

    1) Investing.com（目标：抓取 Name/Value/Action 表格中 RSI(14) 的 Value）：
       - https://www.investing.com/indices/us-spx-500-technical

    2) TradingView（通常动态渲染/反爬更强）：
       - https://www.tradingview.com/symbols/SPX/technicals/

    3) Investtech（仅当页面包含可解析的 RSI(14) 数值时使用）：
       - https://www.investtech.com/main/market.php?CompanyID=10400521&product=211
    """

    logger = logging.getLogger(__name__)

    INVESTTECH_URL = "https://www.investtech.com/main/market.php?CompanyID=10400521&product=211"
    INVESTING_URL = "https://www.investing.com/indices/us-spx-500-technical"
    # Try explicit daily timeframe variants first.
    INVESTING_DAILY_URLS = (
        "https://www.investing.com/indices/us-spx-500-technical?timeFrame=day",
        "https://www.investing.com/indices/us-spx-500-technical?interval=1day",
        INVESTING_URL,
    )
    TRADINGVIEW_URL = "https://www.tradingview.com/symbols/SPX/technicals/?interval=1D"

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()

    def fetch(self, indicator_ids: list[IndicatorId]) -> list[Observation]:
        if indicator_ids and IndicatorId.SP500_RSI not in set(indicator_ids):
            return []
        obs = self._fetch_best_effort()
        return [obs] if obs else []

    def _fetch_best_effort(self) -> Observation | None:
        attempts: list[dict[str, Any]] = []
        for source, fn in (
            ("Investing.com", self._fetch_investing),
            ("TradingView", self._fetch_tradingview),
            ("Investtech", self._fetch_investtech),
        ):
            try:
                obs = fn()
            except Exception as exc:
                attempts.append({"source": source, "timeframe": "1D", "ok": False, "reason": f"exception:{exc}"})
                self.logger.warning("SP500 RSI fetch failed: %s", attempts[-1])
                continue
            if obs:
                self.logger.debug("SP500 RSI fetch succeeded: %s", obs.meta)
                return obs
            attempts.append({"source": source, "timeframe": "1D", "ok": False, "reason": "no_live_rsi14_value"})
            self.logger.info("SP500 RSI fetch no data: %s", attempts[-1])

        self.logger.warning("SP500 RSI all live sources failed: %s", attempts)
        return None

    def _get(self, url: str, *, referer: str | None = None) -> str:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        if referer:
            headers["Referer"] = referer
        try:
            resp = self.session.get(url, headers=headers, timeout=(5, 12))
            if resp.status_code < 400:
                return resp.text or ""
        except Exception:
            resp = None

        # Cloud fallback: some deployments run behind outbound proxy/WAF and can get
        # blocked for Investing while local machine still works. Retry once bypassing
        # env proxy settings.
        if "investing.com" in url:
            try:
                direct = requests.Session()
                direct.trust_env = False
                resp2 = direct.get(url, headers=headers, timeout=(5, 12))
                if resp2.status_code < 400:
                    return resp2.text or ""
            except Exception:
                pass

        return ""

    def _parse_investtech_rsi14_value(self, html: str) -> tuple[float | None, dict[str, str]]:
        if not html:
            return (None, {"match": "empty_html"})

        # Investtech 只接受明确 RSI(14) 的当前数值，不抓泛化评论数字。
        patterns: list[tuple[str, str]] = [
            ("it_rsi14_label_number", r"RSI\s*\(\s*14\s*\)[^0-9]{0,80}([0-9]{1,3}(?:\.[0-9]+)?)"),
            (
                "it_relative_strength_index_14",
                r"Relative\s+Strength\s+Index\s*\(\s*14\s*\)[^0-9]{0,80}([0-9]{1,3}(?:\.[0-9]+)?)",
            ),
            (
                "it_rsi_relative_strength_index",
                r"RSI\s*-\s*Relative\s+Strength\s+Index[^0-9]{0,120}([0-9]{1,3}(?:\.[0-9]+)?)",
            ),
        ]
        for tag, p in patterns:
            m = re.search(p, html, re.IGNORECASE | re.DOTALL)
            if m:
                try:
                    v = float(m.group(1))
                except Exception:
                    continue
                if 0 <= v <= 100:
                    return (v, {"match": tag, "value_raw": m.group(1)})
        return (None, {"match": "no_rsi14_value"})

    def _parse_tradingview_rsi14_value(self, html: str) -> tuple[float | None, dict[str, str]]:
        """
        TradingView: strictly match the RSI(14) row/value, avoid broad numeric grabs.
        Returns (value, debug_meta).
        """
        if not html:
            return (None, {"tv_match": "empty_html"})

        patterns: list[tuple[str, str]] = [
            (
                "tv_oscillators_rsi14_row",
                r"Oscillators.{0,1500}?Relative\s+Strength\s+Index\s*\(\s*14\s*\)\s*"
                r"</[^>]+>\s*"
                r"<[^>]+>\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*</[^>]+>",
            ),
            (
                "tv_row_td_exact",
                r"Relative\s+Strength\s+Index\s*\(\s*14\s*\)\s*"
                r"</[^>]+>\s*"
                r"<[^>]+>\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*</[^>]+>",
            ),
            (
                "tv_row_rsi14_td",
                r"RSI\s*\(\s*14\s*\)\s*"
                r"</[^>]+>\s*"
                r"<[^>]+>\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*</[^>]+>",
            ),
            (
                "tv_json_title_value",
                r"Relative\s+Strength\s+Index\s*\(\s*14\s*\).*?"
                r"(?:\"value\"|value)\s*[:=]\s*\"?([0-9]{1,3}(?:\.[0-9]+)?)\"?",
            ),
            (
                "tv_json_name_value",
                r"(?:\"name\"|name)\s*[:=]\s*\"?Relative\s+Strength\s+Index\s*\(\s*14\s*\)\"?.*?"
                r"(?:\"value\"|value)\s*[:=]\s*\"?([0-9]{1,3}(?:\.[0-9]+)?)\"?",
            ),
        ]

        for tag, p in patterns:
            m = re.search(p, html, re.IGNORECASE | re.DOTALL)
            if not m:
                continue
            try:
                v = float(m.group(1))
            except Exception:
                continue
            if 0 <= v <= 100:
                return (v, {"tv_match": tag, "tv_value_raw": m.group(1)})

        return (None, {"tv_match": "no_rsi14_row_match"})

    def _parse_investing_rsi14_value(self, html: str) -> tuple[float | None, dict[str, str]]:
        """
        解析 Investing.com 技术面 “Name / Value / Action” 表格里的 RSI(14) → Value。

        目标形态（示例）：
        Name        Value     Action
        RSI(14)     69.858    Buy
        """
        if not html:
            return (None, {"match": "empty_html"})

        # 优先：匹配表格整行，提取 RSI(14) 后第一列 value。
        # 只要求存在第三列（action）但不强限制其文本，避免页面把 Buy/Sell
        # 改成 Strong Buy / Strong Sell 后导致解析失效。
        # 典型结构：
        # <td>RSI(14)</td><td>69.858</td><td>Strong Buy</td>
        m = re.search(
            r"RSI\s*\(\s*14\s*\)\s*"
            r"</td>\s*"
            r"<td[^>]*>\s*"
            r"([0-9]{1,3}(?:\.[0-9]+)?)\s*"
            r"</td>\s*"
            r"<td[^>]*>.*?</td>",
            html,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            v = float(m.group(1))
            if 0 <= v <= 100:
                return (v, {"match": "investing_td_exact", "value_raw": m.group(1)})

        # 兜底：有些情况下 td 里会包一层 span/div
        m = re.search(
            r"RSI\s*\(\s*14\s*\)\s*"
            r"</td>\s*"
            r"<td[^>]*>.*?"
            r"([0-9]{1,3}(?:\.[0-9]+)?)"
            r".*?</td>\s*"
            r"<td[^>]*>.*?</td>",
            html,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            v = float(m.group(1))
            if 0 <= v <= 100:
                return (v, {"match": "investing_td_nested", "value_raw": m.group(1)})

        # 再兜底：如果页面把表格数据塞在脚本 JSON 里（key/value/action）
        m = re.search(
            r"RSI\s*\(\s*14\s*\).*?"
            r"(?:\"value\"|value|data-value)\s*[:=]\s*\"?([0-9]{1,3}(?:\.[0-9]+)?)\"?",
            html,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            v = float(m.group(1))
            if 0 <= v <= 100:
                return (v, {"match": "investing_json_value", "value_raw": m.group(1)})

        return (None, {"match": "no_rsi14_row_match"})

    def _fetch_investtech(self) -> Observation | None:
        html = self._get(self.INVESTTECH_URL, referer="https://www.investtech.com/")
        parsed_ok = bool(html)
        v, dbg = self._parse_investtech_rsi14_value(html)
        if v is None:
            self.logger.info(
                "SP500 RSI Investtech parse failed: %s",
                {
                    "source": "Investtech",
                    "timeframe": "1D",
                    "html_ok": parsed_ok,
                    "parse_ok": False,
                    "reason": dbg.get("match"),
                },
            )
            return None
        return Observation(
            indicator_id=IndicatorId.SP500_RSI,
            as_of=date.today(),
            value=v,
            unit="0-100",
            source="Investtech",
            meta={
                "source": "Investtech",
                "url": self.INVESTTECH_URL,
                "timeframe": "1D",
                "html_ok": parsed_ok,
                "parse_ok": True,
                "selector": "RSI(14)->Value",
                **dbg,
            },
        )

    def _fetch_investing(self) -> Observation | None:
        for u in self.INVESTING_DAILY_URLS:
            html = self._get(u, referer="https://www.investing.com/")
            # 只接受 RSI(14) 的 Value
            v, dbg = self._parse_investing_rsi14_value(html)
            if v is None:
                self.logger.info(
                    "SP500 RSI Investing parse failed: %s",
                    {
                        "source": "Investing.com",
                        "timeframe": "1D",
                        "url": u,
                        "html_ok": bool(html),
                        "parse_ok": False,
                        "reason": dbg.get("match"),
                    },
                )
                continue
            return Observation(
                indicator_id=IndicatorId.SP500_RSI,
                as_of=date.today(),
                value=v,
                unit="0-100",
                source="Investing.com",
                meta={
                    "source": "Investing.com",
                    "url": u,
                    "timeframe": "1D",
                    "html_ok": bool(html),
                    "parse_ok": True,
                    "selector": "RSI(14)->Value",
                    **dbg,
                },
            )
        return None

    def _fetch_tradingview(self) -> Observation | None:
        html = self._get(self.TRADINGVIEW_URL, referer="https://www.tradingview.com/")
        v, dbg = self._parse_tradingview_rsi14_value(html)
        if v is None:
            self.logger.info(
                "SP500 RSI TradingView parse failed: %s",
                {
                    "source": "TradingView",
                    "timeframe": "1D",
                    "html_ok": bool(html),
                    "parse_ok": False,
                    "reason": dbg.get("tv_match"),
                },
            )
            return None
        return Observation(
            indicator_id=IndicatorId.SP500_RSI,
            as_of=date.today(),
            value=v,
            unit="0-100",
            source="TradingView",
            meta={
                "source": "TradingView",
                "url": self.TRADINGVIEW_URL,
                "timeframe": "1D",
                "html_ok": bool(html),
                "parse_ok": True,
                "selector": "Relative Strength Index (14)->Value",
                **dbg,
            },
        )
