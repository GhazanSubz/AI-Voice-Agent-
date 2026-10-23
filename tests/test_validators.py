from datetime import date

import pytest

from app.domain import validators as v


@pytest.mark.parametrize(
    "raw, expected",
    [("(415) 555-0132", "4155550132"), ("+1 415 555 0132", "4155550132"), ("14155550132", "4155550132")],
)
def test_phone_normalizes(raw, expected):
    assert v.normalize_us_phone(raw) == expected


@pytest.mark.parametrize("raw", ["555", "415555013", "0155550132", "4151550132", "24155550132"])
def test_phone_rejects(raw):
    with pytest.raises(ValueError):
        v.normalize_us_phone(raw)


def test_dob_formats_and_bounds():
    today = date(2026, 10, 23)
    assert v.parse_date_of_birth("03/05/1982", today) == date(1982, 3, 5)
    assert v.parse_date_of_birth("1982-03-05", today) == date(1982, 3, 5)
    assert v.parse_date_of_birth("10/23/2026", today) == today
    for bad in ["10/24/2026", "02/30/1990", "1899-12-31", "yesterday"]:
        with pytest.raises(ValueError):
            v.parse_date_of_birth(bad, today)


@pytest.mark.parametrize("name", ["Jane", "O'Neil", "Mary-Kate", "De La Cruz", "José"])
def test_names_accepted(name):
    assert v.validate_name(name) == name


@pytest.mark.parametrize("name", ["", "J4ne", "Jane!", "-Jane", "A" * 51])
def test_names_rejected(name):
    with pytest.raises(ValueError):
        v.validate_name(name)


def test_state_accepts_code_or_name():
    assert v.normalize_state("tx") == "TX"
    assert v.normalize_state("New York") == "NY"
    assert v.normalize_state("D.C.") == "DC"
    with pytest.raises(ValueError):
        v.normalize_state("ZZ")


def test_zip_formats():
    assert v.normalize_zip("94105") == "94105"
    assert v.normalize_zip("941051234") == "94105-1234"
    assert v.normalize_zip("94105-1234") == "94105-1234"
    for bad in ["9410", "94105-12", "ABCDE"]:
        with pytest.raises(ValueError):
            v.normalize_zip(bad)


def test_sex_aliases():
    assert v.normalize_sex("f") == "Female"
    assert v.normalize_sex("Prefer not to say") == "Decline to Answer"
    with pytest.raises(ValueError):
        v.normalize_sex("unknown")


def test_member_id():
    assert v.normalize_member_id("xkb 123-456") == "XKB123456"
    with pytest.raises(ValueError):
        v.normalize_member_id("ABC#1")


def test_clean_text_strips_control_chars():
    assert v.clean_text("  Ja\x00ne \n Doe ") == "Jane Doe"
