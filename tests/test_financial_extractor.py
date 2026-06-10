"""
Unit tests for src/utils/financial_extractor.py

Tests pure utility functions — no API calls.
"""

import pytest
from src.utils.financial_extractor import (
    fix_units,
    clean_invalid_values,
    calculate_yoy_changes,
    parse_llm_json,
)


class TestFixUnits:

    def _make_data(self, revenue):
        return {
            "income_statement": {
                "total_net_sales":  {"year_1": revenue, "year_2": None, "year_3": None},
                "net_income":       {"year_1": 112.0,   "year_2": None, "year_3": None},
                "eps_diluted":      {"year_1": 6.75,    "year_2": None, "year_3": None},
            },
            "operating_expenses": {},
            "product_segments": {}
        }

    def test_billions_converted_to_millions(self):
        """Values under 10,000 (billions) should be multiplied by 1000."""
        data   = self._make_data(revenue=416.161)
        result = fix_units(data)
        sales  = result["income_statement"]["total_net_sales"]["year_1"]
        assert sales == 416161.0

    def test_millions_unchanged(self):
        """Values already in millions (>10,000) should not be changed."""
        data   = self._make_data(revenue=416161.0)
        result = fix_units(data)
        sales  = result["income_statement"]["total_net_sales"]["year_1"]
        assert sales == 416161.0

    def test_eps_not_multiplied(self):
        """EPS values should never be multiplied — they are per-share."""
        data   = self._make_data(revenue=416.0)
        result = fix_units(data)
        eps    = result["income_statement"]["eps_diluted"]["year_1"]
        assert eps == 6.75   # unchanged


class TestCleanInvalidValues:

    def test_negative_revenue_replaced_with_none(self):
        data = {
            "income_statement": {
                "total_net_sales": {"year_1": 416161, "year_2": -4, "year_3": 383285}
            },
            "operating_expenses": {},
            "product_segments": {}
        }
        result = clean_invalid_values(data)
        assert result["income_statement"]["total_net_sales"]["year_2"] is None

    def test_valid_values_unchanged(self):
        data = {
            "income_statement": {
                "total_net_sales": {"year_1": 416161, "year_2": 391035, "year_3": 383285}
            },
            "operating_expenses": {},
            "product_segments": {}
        }
        result = clean_invalid_values(data)
        assert result["income_statement"]["total_net_sales"]["year_1"] == 416161

    def test_none_values_remain_none(self):
        data = {
            "income_statement": {
                "total_net_sales": {"year_1": None, "year_2": None, "year_3": None}
            },
            "operating_expenses": {},
            "product_segments": {}
        }
        result = clean_invalid_values(data)
        assert result["income_statement"]["total_net_sales"]["year_1"] is None


class TestCalculateYoyChanges:

    def test_positive_growth_detected(self):
        data = {
            "income_statement": {
                "total_net_sales": {
                    "year_1": 416161,
                    "year_2": 391035,
                    "year_3": 383285
                }
            },
            "operating_expenses": {},
            "product_segments": {}
        }
        result = calculate_yoy_changes(data)
        yoy    = result["yoy_changes"]["income_statement"]["total_net_sales"]
        assert yoy["yoy_change_pct"] > 0
        assert yoy["direction"] == "▲"

    def test_yoy_math_correct(self):
        """YoY% = ((new - old) / abs(old)) * 100"""
        data = {
            "income_statement": {
                "net_income": {"year_1": 112010, "year_2": 93736, "year_3": 96995}
            },
            "operating_expenses": {},
            "product_segments": {}
        }
        result   = calculate_yoy_changes(data)
        yoy_pct  = result["yoy_changes"]["income_statement"]["net_income"]["yoy_change_pct"]
        expected = round(((112010 - 93736) / abs(93736)) * 100, 2)
        assert abs(yoy_pct - expected) < 0.01

    def test_none_values_return_none_yoy(self):
        data = {
            "income_statement": {
                "eps_diluted": {"year_1": None, "year_2": None, "year_3": None}
            },
            "operating_expenses": {},
            "product_segments": {}
        }
        result = calculate_yoy_changes(data)
        yoy    = result["yoy_changes"]["income_statement"]["eps_diluted"]
        assert yoy["yoy_change_pct"] is None


class TestParseLlmJson:

    def test_parses_raw_json(self):
        raw    = '{"company": "Apple Inc.", "ticker": "AAPL"}'
        result = parse_llm_json(raw)
        assert result["ticker"] == "AAPL"

    def test_strips_markdown_fences(self):
        raw    = '```json\n{"ticker": "AAPL"}\n```'
        result = parse_llm_json(raw)
        assert result["ticker"] == "AAPL"

    def test_strips_plain_code_fence(self):
        raw    = '```\n{"ticker": "NVDA"}\n```'
        result = parse_llm_json(raw)
        assert result["ticker"] == "NVDA"

    def test_handles_surrounding_text(self):
        raw    = 'Here is the data: {"ticker": "AAPL"} end.'
        result = parse_llm_json(raw)
        assert result["ticker"] == "AAPL"

    def test_invalid_json_raises(self):
        with pytest.raises((ValueError, Exception)):
            parse_llm_json("not json at all")