"""
Unit tests for src/utils/sentiment_analyzer.py

Tests lexicon scoring and phrase extraction — no API calls.
"""

import pytest
from src.utils.sentiment_analyzer import (
    analyze_lexicon,
    extract_sentiment_phrases,
)


class TestAnalyzeLexicon:

    def test_positive_text_scores_positive(self):
        text   = ("Revenue growth exceeded expectations. "
                  "Strong performance across all segments. "
                  "Record profitability achieved this quarter. " * 5)
        result = analyze_lexicon(text)
        assert result["net_sentiment_score"] > 0
        assert result["sentiment_label"] == "POSITIVE"

    def test_negative_text_scores_negative(self):
        text   = ("Revenue declined significantly. "
                  "Challenging macro conditions created difficult headwinds. "
                  "Loss increased due to adverse market volatility. " * 5)
        result = analyze_lexicon(text)
        assert result["net_sentiment_score"] < 0

    def test_word_counts_are_non_negative(self):
        text   = "Apple reported results for fiscal year 2024."
        result = analyze_lexicon(text)
        assert result["positive_count"]        >= 0
        assert result["negative_count"]        >= 0
        assert result["uncertainty_count"]     >= 0
        assert result["litigious_count"]       >= 0
        assert result["forward_looking_count"] >= 0

    def test_word_count_matches_input(self):
        text   = "revenue profit growth increase strong record"
        result = analyze_lexicon(text)
        assert result["word_count"] == 6

    def test_empty_text_returns_zeros(self):
        result = analyze_lexicon("")
        assert result["positive_count"]      == 0
        assert result["negative_count"]      == 0
        assert result["net_sentiment_score"] == 0.0

    def test_uncertainty_words_counted(self):
        text   = "We may expect revenue could possibly increase if conditions improve."
        result = analyze_lexicon(text)
        assert result["uncertainty_count"] > 0

    def test_forward_looking_counted(self):
        text   = "We expect to deliver strong growth and will invest in future expansion."
        result = analyze_lexicon(text)
        assert result["forward_looking_count"] > 0


class TestExtractSentimentPhrases:

    def test_returns_dict_with_expected_keys(self):
        text   = "Revenue growth exceeded expectations this year."
        result = extract_sentiment_phrases(text)
        assert "top_positive_sentences" in result
        assert "top_negative_sentences" in result

    def test_positive_sentences_found(self):
        text = (
            "Revenue growth exceeded expectations significantly. "
            "Strong performance across all product lines improved results. "
            "Record profitability achieved with outstanding growth momentum. "
            "The company delivered exceptional results in all markets. "
        )
        result = extract_sentiment_phrases(text)
        assert len(result["top_positive_sentences"]) > 0

    def test_negative_sentences_found(self):
        text = (
            "Revenue declined sharply amid challenging macroeconomic conditions. "
            "Difficult headwinds created adverse pressure on gross margins. "
            "Loss increased due to volatile market conditions and uncertainty. "
        )
        result = extract_sentiment_phrases(text)
        assert len(result["top_negative_sentences"]) > 0

    def test_very_short_sentences_excluded(self):
        text   = "Up. Down. Revenue. Loss. " * 10
        result = extract_sentiment_phrases(text)
        for sent in result["top_positive_sentences"] + result["top_negative_sentences"]:
            assert len(sent) >= 30

    def test_empty_text_returns_empty_lists(self):
        result = extract_sentiment_phrases("")
        assert result["top_positive_sentences"] == []
        assert result["top_negative_sentences"] == []