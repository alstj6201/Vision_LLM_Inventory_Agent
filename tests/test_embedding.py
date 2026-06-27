import pytest

from src.embedding import parse_image_filename, parse_product_folder


def test_parse_product_folder_keeps_product_name_underscores():
    sku_id, product_name = parse_product_folder("46019_아이배냇)쌩마멧유기농사과_바나나")

    assert sku_id == "46019"
    assert product_name == "아이배냇)쌩마멧유기농사과_바나나"


def test_parse_image_filename_extracts_height_and_angle():
    sku_id, height, angle = parse_image_filename("10060_30_s_11.jpg")

    assert sku_id == "10060"
    assert height == 30
    assert angle == 11


def test_parse_image_filename_accepts_alphanumeric_sku():
    sku_id, height, angle = parse_image_filename("A30035_00_s_22.jpg")

    assert sku_id == "A30035"
    assert height == 0
    assert angle == 22


def test_parse_image_filename_rejects_non_s_view_type():
    with pytest.raises(ValueError):
        parse_image_filename("55845_00_m_4.jpg")
