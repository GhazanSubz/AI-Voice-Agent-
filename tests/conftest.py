import logging
import os

# Never load the TTS model in tests.
os.environ["TTS_WARM_UP"] = "false"
os.environ["LOG_FILE"] = ""  # keep test runs out of logs/app.log

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture(autouse=True)
def _log_everything(caplog):
    # Run every log call for real: a bad `extra=` key only fails when the record is built,
    # which never happens if the level filters it out (this hid a crash on the first live call).
    caplog.set_level(logging.DEBUG)


@pytest.fixture
def db_url(tmp_path):
    return f"sqlite:///{tmp_path / 'test.db'}"


@pytest.fixture
def client(db_url):
    app = create_app(Settings(database_url=db_url, seed_demo_data=False, log_level="DEBUG"))
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def valid_patient():
    return {
        "first_name": "Maria",
        "last_name": "Davis",
        "date_of_birth": "07/04/1990",
        "sex": "Female",
        "phone_number": "(312) 555-0110",
        "address_line_1": "42 Lake Shore Drive",
        "city": "Chicago",
        "state": "IL",
        "zip_code": "60611",
    }
