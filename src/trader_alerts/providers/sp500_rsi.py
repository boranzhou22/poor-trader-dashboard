from __future__ import annotations

import re
from datetime import date

import requests

from ..constants import IndicatorId
from ..models import Observation
from .base import Provider


class Sp500RsiProvider(Provider):
    """
    S&P 500 RSI 数据源（按“可解析到 RSI(14) Value”的顺序尝试）：

    1) Investing.com（目标：抓取 Name/Value/Action 表格中 RSI(14) 的 Value）：
       - https://www.investing.com/indices/us-spx-500-technical

    2) Investtech（有时仅有文字描述，不一定能拿到“RSI 数值”；作为兜底）：
       - https://www.investtech.com/main/market.php?CompanyID=10400521&product=211

    3) TradingView（通常动态渲染/反爬更强）：
       - https://www.tradingview.com/symbols/SPX/technicals/
    """

    INVESTTECH_URL = "https://www.investtech.com/main/market.php?CompanyID=10400521&product=211"
    INVESTING_URL = "https://www.investing.com/indices/us-spx-500-technical"
    # Try explicit daily timeframe variants first.
    INVESTING_DAILY_URLS = (
        "https://www.investing.com/indices/us-spx-500-technical?timeFrame=day",
        "https://www.investing.com/indices/us-spx-500-technical?interval=1day",
        INVESTING_URL,
    )
    TRADINGVIEW_URL = "https://www.tradingview.com/symbols/SPX/technicals/?interval=1D"
    STOOQ_DAILY_URL = "https://stooq.com/q/d/l/"

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()

    def fetch(self, indicator_ids: list[IndicatorId]) -> list[Observation]:
        if indicator_ids and IndicatorId.SP500_RSI not in set(indicator_ids):
            return []
        obs = self._fetch_best_effort()
        return [obs] if obs else []

    def _fetch_best_effort(self) -> Observation | None:
        # Requested source order (daily RSI intent):
        # 1) Investing.com
        # 2) Investtech
        # 3) TradingView
        # 4) Stooq daily calc (final emergency fallback only)
        for fn in (self._fetch_investing, self._fetch_investtech, self._fetch_tradingview, self._fetch_stooq_rsi):
            try:
                obs = fn()
            except Exception:
                obs = None
            if obs is not None:
                return obs
        return None

    def _compute_rsi14(self, closes: list[float]) -> float | None:
        # Wilder RSI(14). Need at least 15 closes.
        if len(closes) < 15:
            return None
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [max(d, 0.0) for d in deltas]
        losses = [max(-d, 0.0) for d in deltas]

        period = 14
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        if 0 <= rsi <= 100:
            return rsi
        return None

    def _fetch_stooq_rsi(self) -> Observation | None:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv,text/plain,*/*",
            "Accept-Encoding": "identity",
        }
        params = {"s": "^spx", "i": "d"}
        resp = self.session.get(self.STOOQ_DAILY_URL, params=params, headers=headers, timeout=(5, 12))
        if resp.status_code >= 400:
            return None
        text = (resp.text or "").strip()
        if not text or text.lower().startswith("no data"):
            return None
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [max(d, 0.0) for d in deltas]
        losses = [max(-d, 0.0) for d in deltas]

        period = 14
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        if 0 <= rsi <= 100:
            return rsi
        return None

    def _fetch_stooq_rsi(self) -> Observation | None:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv,text/plain,*/*",
            "Accept-Encoding": "identity",
        }
        params = {"s": "^spx", "i": "d"}
        resp = self.session.get(self.STOOQ_DAILY_URL, params=params, headers=headers, timeout=(5, 12))
        if resp.status_code >= 400:
            return None
        text = (resp.text or "").strip()
        if not text or text.lower().startswith("no data"):
            return None

        closes: list[float] = []
        last_as_of: date | None = None
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) < 5:
                continue
            try:
                d = date.fromisoformat(parts[0])
                c = float(parts[4])
            except Exception:
                continue
            last_as_of = d
            closes.append(c)

        if not last_as_of:
            return None
        rsi = self._compute_rsi14(closes)
        if rsi is None:
            return None

        return Observation(
            indicator_id=IndicatorId.SP500_RSI,
            as_of=last_as_of,
            value=rsi,
            unit="0-100",
            source="Stooq(calc)",
            meta={"url": self.STOOQ_DAILY_URL, "symbol": "^spx", "method": "wilder_rsi14"},
        )

        closes: list[float] = []
        last_as_of: date | None = None
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) < 5:
                continue
            try:
                d = date.fromisoformat(parts[0])
                c = float(parts[4])
            except Exception:
                continue
            last_as_of = d
            closes.append(c)

        if not last_as_of:
            return None
        rsi = self._compute_rsi14(closes)
        if rsi is None:
            return None

        return Observation(
            indicator_id=IndicatorId.SP500_RSI,
            as_of=last_as_of,
            value=rsi,
            unit="0-100",
            source="Stooq(calc)",
            meta={"url": self.STOOQ_DAILY_URL, "symbol": "^spx", "method": "wilder_rsi14"},
        )

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

    def _parse_rsi_from_html(self, html: str) -> float | None:
        if not html:
            return None

        # 常见形式（不同站点可能会出现的 RSI(14) / Relative Strength Index (14) / RSI - Relative Strength Index）
        # 注意：不要使用过宽的 "\bRSI\b ... number" 规则，避免误抓页面其它数字（云端更易触发反爬页面）。
        patterns = [
            r"Relative\s+Strength\s+Index\s*\(14\)[^0-9]{0,80}([0-9]{1,3}(?:\.[0-9]+)?)",
            r"RSI\s*\(14\)[^0-9]{0,80}([0-9]{1,3}(?:\.[0-9]+)?)",
            r"RSI\s*-\s*Relative\s+Strength\s+Index[^0-9]{0,120}([0-9]{1,3}(?:\.[0-9]+)?)",
        ]
        for p in patterns:
            m = re.search(p, html, re.IGNORECASE)
            if m:
                v = float(m.group(1))
                if 0 <= v <= 100:
                    return v
        return None

    def _parse_tradingview_rsi14_value(self, html: str) -> tuple[float | None, dict[str, str]]:
        """
        TradingView: strictly match the RSI(14) row/value, avoid broad numeric grabs.
        Returns (value, debug_meta).
        """
        if not html:
            return (None, {"tv_match": "empty_html"})

        patterns: list[tuple[str, str]] = [
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

    def _parse_investing_rsi14_value(self, html: str) -> float | None:
        """
        解析 Investing.com 技术面 “Name / Value / Action” 表格里的 RSI(14) → Value。

        目标形态（示例）：
        Name        Value     Action
        RSI(14)     69.858    Buy
        """
        if not html:
            return None

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
                return v

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
                return v

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
                return v

        # 兜底：抓取 RSI(14) 后 0~200 字符内出现的第一个数值
        m = re.search(r"RSI\s*\(\s*14\s*\)(.{0,200})", html, re.IGNORECASE | re.DOTALL)
        if m:
            mm = re.search(r"([0-9]{1,3}(?:\.[0-9]+)?)", m.group(1))
            if mm:
                v = float(mm.group(1))
                if 0 <= v <= 100:
                    return v

        return None

    def _fetch_investtech(self) -> Observation | None:
        html = self._get(self.INVESTTECH_URL, referer="https://www.investtech.com/")
        v = self._parse_rsi_from_html(html)
        if v is None:
            return None
        return Observation(
            indicator_id=IndicatorId.SP500_RSI,
            as_of=date.today(),
            value=v,
            unit="0-100",
            source="Investtech",
            meta={"url": self.INVESTTECH_URL, "timeframe": "1D"},
        )

    def _fetch_investing(self) -> Observation | None:
        for u in self.INVESTING_DAILY_URLS:
            html = self._get(u, referer="https://www.investing.com/")
            # 只接受 RSI(14) 的 Value
            v = self._parse_investing_rsi14_value(html)
            if v is None:
                continue
            return Observation(
                indicator_id=IndicatorId.SP500_RSI,
                as_of=date.today(),
                value=v,
                unit="0-100",
                source="Investing.com",
                meta={"url": u, "timeframe": "1D"},
            )
        return None

    def _fetch_tradingview(self) -> Observation | None:
        html = self._get(self.TRADINGVIEW_URL, referer="https://www.tradingview.com/")
        v, dbg = self._parse_tradingview_rsi14_value(html)
        if v is None:
            return None
        return Observation(
            indicator_id=IndicatorId.SP500_RSI,
            as_of=date.today(),
            value=v,
            unit="0-100",
            source="TradingView",
            meta={"url": self.TRADINGVIEW_URL, "timeframe": "1D", **dbg},
        )
