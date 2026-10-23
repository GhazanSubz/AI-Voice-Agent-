import uuid

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def create(client, payload):
    response = client.post("/patients", json=payload)
    assert response.status_code == 201, response.text
    return response.json()["data"]


class TestCreate:
    def test_returns_201_with_normalized_record(self, client, valid_patient):
        response = client.post("/patients", json=valid_patient)

        assert response.status_code == 201
        body = response.json()
        assert body["error"] is None
        data = body["data"]
        uuid.UUID(data["patient_id"])
        assert data["phone_number"] == "3125550110"
        assert data["date_of_birth"] == "07/04/1990"
        assert data["preferred_language"] == "English"
        assert data["created_at"].endswith("Z")

    def test_invalid_fields_return_422_with_per_field_messages(self, client, valid_patient):
        payload = {**valid_patient, "phone_number": "555", "date_of_birth": "01/01/2999"}

        response = client.post("/patients", json=payload)

        assert response.status_code == 422
        body = response.json()
        assert body["data"] is None
        fields = {d["field"]: d["message"] for d in body["error"]["details"]}
        assert "10 digits" in fields["phone_number"]
        assert "future" in fields["date_of_birth"]

    def test_missing_required_field_is_422(self, client, valid_patient):
        del valid_patient["last_name"]
        response = client.post("/patients", json=valid_patient)
        assert response.status_code == 422
        assert response.json()["error"]["details"][0]["field"] == "last_name"

    def test_unknown_field_is_rejected(self, client, valid_patient):
        response = client.post("/patients", json={**valid_patient, "ssn": "123-45-6789"})
        assert response.status_code == 422

    def test_malformed_json_is_400(self, client):
        response = client.post(
            "/patients", content="{not json", headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "bad_request"

    def test_non_object_body_is_400(self, client):
        assert client.post("/patients", json=["a"]).status_code == 400


class TestRead:
    def test_get_by_id(self, client, valid_patient):
        created = create(client, valid_patient)
        response = client.get(f"/patients/{created['patient_id']}")
        assert response.status_code == 200
        assert response.json()["data"] == created

    def test_unknown_id_is_404(self, client):
        response = client.get(f"/patients/{uuid.uuid4()}")
        assert response.status_code == 404
        assert response.json() == {
            "data": None,
            "error": {"code": "not_found", "message": response.json()["error"]["message"], "details": None},
        }

    def test_malformed_id_is_400(self, client):
        assert client.get("/patients/not-a-uuid").status_code == 400

    def test_list_filters(self, client, valid_patient):
        create(client, valid_patient)
        create(client, {**valid_patient, "last_name": "Smith", "phone_number": "3125550111"})

        by_name = client.get("/patients", params={"last_name": "davis"}).json()["data"]
        by_phone = client.get("/patients", params={"phone_number": "312-555-0111"}).json()["data"]
        by_dob = client.get("/patients", params={"date_of_birth": "1990-07-04"}).json()["data"]

        assert [p["last_name"] for p in by_name] == ["Davis"]
        assert [p["last_name"] for p in by_phone] == ["Smith"]
        assert len(by_dob) == 2

    def test_invalid_filter_is_400(self, client):
        assert client.get("/patients", params={"date_of_birth": "13/45/2000"}).status_code == 400

    def test_unknown_route_uses_envelope(self, client):
        response = client.get("/nope")
        assert response.status_code == 404
        assert response.json()["data"] is None


class TestUpdate:
    def test_partial_update_changes_only_sent_fields(self, client, valid_patient):
        created = create(client, valid_patient)

        response = client.put(f"/patients/{created['patient_id']}", json={"last_name": "Davies"})

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["last_name"] == "Davies"
        assert data["first_name"] == "Maria"
        assert data["updated_at"] >= created["updated_at"]

    def test_optional_field_can_be_cleared(self, client, valid_patient):
        created = create(client, {**valid_patient, "email": "maria@example.com"})
        response = client.put(f"/patients/{created['patient_id']}", json={"email": None})
        assert response.json()["data"]["email"] is None

    def test_required_field_cannot_be_cleared(self, client, valid_patient):
        created = create(client, valid_patient)
        response = client.put(f"/patients/{created['patient_id']}", json={"city": None})
        assert response.status_code == 422

    def test_empty_update_is_422(self, client, valid_patient):
        created = create(client, valid_patient)
        assert client.put(f"/patients/{created['patient_id']}", json={}).status_code == 422

    def test_update_unknown_is_404(self, client):
        assert client.put(f"/patients/{uuid.uuid4()}", json={"city": "Austin"}).status_code == 404


class TestSoftDelete:
    def test_deleted_patient_is_hidden_but_kept(self, client, valid_patient):
        created = create(client, valid_patient)
        pid = created["patient_id"]

        assert client.delete(f"/patients/{pid}").status_code == 200

        assert client.get(f"/patients/{pid}").status_code == 404
        assert client.get("/patients").json()["data"] == []
        assert client.delete(f"/patients/{pid}").status_code == 404
        assert client.put(f"/patients/{pid}", json={"city": "Austin"}).status_code == 404

        engine = client.app.state.engine
        with engine.connect() as conn:
            row = conn.exec_driver_sql("SELECT deleted_at FROM patients").one()
        assert row.deleted_at is not None


def test_data_survives_app_restart(db_url, valid_patient):
    settings = Settings(database_url=db_url, seed_demo_data=False, log_level="WARNING")
    with TestClient(create_app(settings)) as first:
        pid = create(first, valid_patient)["patient_id"]

    with TestClient(create_app(settings)) as second:
        assert second.get(f"/patients/{pid}").status_code == 200


def test_seed_runs_once(db_url):
    settings = Settings(database_url=db_url, seed_demo_data=True, log_level="WARNING")
    for _ in range(2):
        with TestClient(create_app(settings)) as c:
            assert len(c.get("/patients").json()["data"]) == 2


def test_database_failure_returns_500_envelope(client, valid_patient):
    client.app.state.engine.dispose()
    with client.app.state.engine.connect() as conn:
        conn.exec_driver_sql("DROP TABLE patients")
        conn.commit()

    response = client.post("/patients", json=valid_patient)

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "database_error"


def test_health(client):
    assert client.get("/health").json() == {"data": {"status": "ok"}, "error": None}
