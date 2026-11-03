"""Normalizers and validators for patient fields.

Each function takes loosely formatted input (as typed into the API or transcribed from speech),
returns the canonical stored form, or raises ValueError with a message short and plain enough
for the voice agent to read back to a caller.
"""

import difflib
import re
import unicodedata
from datetime import date, datetime

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")
# Letters (any script, so "José" works), separated by single spaces, hyphens or apostrophes.
_NAME = re.compile(r"^[^\W\d_]+(?:[ '\-][^\W\d_]+)*$")
_ZIP = re.compile(r"^(\d{5})(?:-?(\d{4}))?$")
_MEMBER_ID = re.compile(r"^[A-Z0-9]{1,30}$")
_EARLIEST_DOB = date(1900, 1, 1)

US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "PR": "Puerto Rico", "GU": "Guam", "VI": "U.S. Virgin Islands", "AS": "American Samoa",
    "MP": "Northern Mariana Islands",
}
_STATE_BY_NAME = {name.lower(): code for code, name in US_STATES.items()}

SEX_ALIASES = {
    "male": "Male", "m": "Male", "man": "Male",
    "female": "Female", "f": "Female", "woman": "Female",
    "other": "Other", "non-binary": "Other", "nonbinary": "Other",
    "decline to answer": "Decline to Answer", "decline": "Decline to Answer",
    "prefer not to say": "Decline to Answer", "prefer not to answer": "Decline to Answer",
}


def clean_text(value: object) -> object:
    """Strip control characters and collapse whitespace. Non-strings pass through untouched."""
    if not isinstance(value, str):
        return value
    value = unicodedata.normalize("NFC", value)
    value = _CONTROL_CHARS.sub("", value)
    return _WHITESPACE.sub(" ", value).strip()


def validate_name(value: str, max_len: int = 50) -> str:
    if not value:
        raise ValueError("name is required")
    if len(value) > max_len:
        raise ValueError(f"name must be at most {max_len} characters")
    if not _NAME.match(value):
        raise ValueError("name may only contain letters, hyphens, apostrophes and spaces")
    return value


def parse_date_of_birth(value: object, today: date | None = None) -> date:
    """Accept MM/DD/YYYY (the spec's format), ISO YYYY-MM-DD or a date object."""
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        dob = value
    elif isinstance(value, str):
        for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
            try:
                dob = datetime.strptime(value.strip(), fmt).date()
                break
            except ValueError:
                continue
        else:
            raise ValueError("date of birth must be a real date in MM/DD/YYYY format")
    else:
        raise ValueError("date of birth must be a real date in MM/DD/YYYY format")

    today = today or date.today()
    if dob > today:
        raise ValueError("date of birth cannot be in the future")
    if dob < _EARLIEST_DOB:
        raise ValueError("date of birth must be after 1900")
    return dob


def normalize_sex(value: str) -> str:
    canonical = SEX_ALIASES.get(value.strip().lower())
    if canonical is None:
        raise ValueError("sex must be Male, Female, Other, or Decline to Answer")
    return canonical


def normalize_us_phone(value: str) -> str:
    """Return a 10-digit NANP number, accepting punctuation and a leading +1/1."""
    digits = re.sub(r"\D", "", value)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        raise ValueError(f"phone number must have 10 digits, got {len(digits)}")
    # NANP: neither the area code nor the exchange may start with 0 or 1.
    if digits[0] in "01" or digits[3] in "01":
        raise ValueError("that is not a valid U.S. phone number")
    return digits


def normalize_state(value: str) -> str:
    code = value.strip().upper().replace(".", "")
    if code in US_STATES:
        return code
    by_name = _STATE_BY_NAME.get(value.strip().lower())
    if by_name:
        return by_name
    raise ValueError("state must be a valid U.S. state or its 2-letter abbreviation")


# USPS ZIP prefixes: the first three digits of a ZIP code identify its state. Ranges are inclusive.
# Military (AA/AE/AP) prefixes are deliberately absent, so they map to no state.
_ZIP3_RANGES = (
    (5, 5, "NY"), (6, 7, "PR"), (8, 8, "VI"), (9, 9, "PR"), (10, 27, "MA"), (28, 29, "RI"),
    (30, 38, "NH"), (39, 49, "ME"), (50, 54, "VT"), (55, 55, "MA"), (56, 59, "VT"), (60, 69, "CT"),
    (70, 89, "NJ"), (100, 149, "NY"), (150, 196, "PA"), (197, 199, "DE"), (200, 200, "DC"),
    (201, 201, "VA"), (202, 205, "DC"), (206, 219, "MD"), (220, 246, "VA"), (247, 268, "WV"),
    (270, 289, "NC"), (290, 299, "SC"), (300, 319, "GA"), (320, 339, "FL"), (341, 349, "FL"),
    (350, 369, "AL"), (370, 385, "TN"), (386, 397, "MS"), (398, 399, "GA"), (400, 427, "KY"),
    (430, 459, "OH"), (460, 479, "IN"), (480, 499, "MI"), (500, 528, "IA"), (530, 549, "WI"),
    (550, 567, "MN"), (569, 569, "DC"), (570, 577, "SD"), (580, 588, "ND"), (590, 599, "MT"),
    (600, 629, "IL"), (630, 658, "MO"), (660, 679, "KS"), (680, 693, "NE"), (700, 714, "LA"),
    (716, 729, "AR"), (730, 732, "OK"), (733, 733, "TX"), (734, 749, "OK"), (750, 799, "TX"),
    (800, 816, "CO"), (820, 831, "WY"), (832, 838, "ID"), (840, 847, "UT"), (850, 865, "AZ"),
    (870, 884, "NM"), (885, 885, "TX"), (889, 898, "NV"), (900, 961, "CA"), (967, 968, "HI"),
    (969, 969, "GU"), (970, 979, "OR"), (980, 994, "WA"), (995, 999, "AK"),
)


def state_for_zip(zip_code: str) -> str | None:
    """2-letter state for a ZIP code, from its first three digits, or None if unknown."""
    if not re.match(r"^\d{5}", zip_code or ""):
        return None
    prefix = int(zip_code[:3])
    for low, high, state in _ZIP3_RANGES:
        if low <= prefix <= high:
            return state
    return None


def suggest_states(heard: str, limit: int = 2) -> list[str]:
    """Closest real state names to a misheard one ("Delivery" -> Delaware), best first."""
    names = {name.lower(): name for name in US_STATES.values()}
    matches = difflib.get_close_matches(heard.strip().lower(), names, n=limit, cutoff=0.5)
    return [names[m] for m in matches]


def normalize_zip(value: str) -> str:
    match = _ZIP.match(value.replace(" ", ""))
    if not match:
        raise ValueError("ZIP code must be 5 digits or ZIP+4")
    five, plus4 = match.groups()
    return f"{five}-{plus4}" if plus4 else five


def normalize_member_id(value: str) -> str:
    # Spoken and printed IDs arrive with spaces or dashes ("ABC 123-456"); store the bare characters.
    compact = re.sub(r"[\s\-]", "", value).upper()
    if not _MEMBER_ID.match(compact):
        raise ValueError("member ID may only contain letters and numbers (up to 30)")
    return compact


def require_text(value: str, label: str, max_len: int) -> str:
    if not value:
        raise ValueError(f"{label} is required")
    if len(value) > max_len:
        raise ValueError(f"{label} must be at most {max_len} characters")
    if not any(ch.isalnum() for ch in value):
        raise ValueError(f"{label} must contain letters or numbers")
    return value
