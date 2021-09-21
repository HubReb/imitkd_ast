#!/usr/bin/env python3
# author: R. Hubert
# # email: hubert@cl.uni-heidelberg.de

import pytest

from preprocess.change_europarl_format import change_format


def test_change_format_correct_input():
    test_list = ["source sentence\t target sentence"]
    source, target = change_format(test_list)
    assert source == ["source sentence"]
    assert target == ["target sentence"]


def test_change_format_correct_input_no_input():
    with pytest.raises(IndexError):
        change_format([])


def test_change_format_correct_input_wrong_type():
    with pytest.raises(TypeError):
        change_format("only string")


def test_change_format_correct_input_empty():
    with pytest.raises(AssertionError):
        change_format(["there is no tab here"])


def test_change_format_multiple_tabs():
    test_list = ["source sentence\t\t\tnewword\t target sentence"]
    with pytest.raises(ValueError):
        change_format(test_list)


def test_change_format_multiple_tabs_starts_hack():
    test_list = ["12.\tsource sentence newword\t target sentence"]
    source, target = change_format(test_list)
    assert source == ["12. source sentence newword"]
    assert target == ["target sentence"]
