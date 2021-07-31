#!/usr/bin/env python3
# author: R. Hubert
# # email: hubert@cl.uni-heidelberg.de

import pytest

from preprocess.change_news_format import change_format_news


def test_change_format_news_correct_input():
    test_list = ["source sentence\t target sentence"]
    source, target = change_format_news(test_list)
    assert source == ["source sentence"]
    assert target == ["target sentence"]


def test_change_format_news_correct_input_no_input():
    with pytest.raises(IndexError):
        change_format_news([])


def test_change_format_news_correct_input_wrong_type():
    with pytest.raises(TypeError):
        change_format_news("only string")


def test_change_format_news_correct_input_empty():
    with pytest.raises(AssertionError):
        change_format_news(["there is no tab here"])


def test_change_format_news_multiple_tabs():
    test_list = ["source sentence\t\t\tnewword\t target sentence"]
    source, target = change_format_news(test_list)
    assert source == ["source sentence"]
    assert target == ["newword target sentence"]


def test_change_format_news_multiple_tabs_starts_hack():
    test_list = ["12.\tsource sentence newword\t target sentence"]
    source, target = change_format_news(test_list)
    assert source == ["12. source sentence newword"]
    assert target == ["target sentence"]
