"""Dashboard page and the /calls endpoint it reads from.

The page renders in the browser from the REST API, so these tests pin the page's contract (it
calls the API, never uses innerHTML) and the API responses it depends on.
"""

import uuid


def test_root_redirects_to_dashboard(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/dashboard"


def test_dashboard_loads_data_from_the_rest_api(client):
    page = client.get("/dashboard").text
    assert 'api("/patients?limit=500")' in page and 'api("/calls?limit=30")' in page
    assert all(name in page for name in ("last_name", "phone_number", "date_of_birth"))


def test_dashboard_never_injects_html(client):
    # Names and transcripts come from callers; rendering must go through text nodes only.
    assert "innerHTML" not in client.get("/dashboard").text


def test_search_errors_come_back_in_the_envelope_the_page_displays(client):
    body = client.get("/patients", params={"phone_number": "555"}).json()
    assert body["data"] is None
    assert "10 digits" in body["error"]["details"][0]["message"]


def test_calls_endpoint_lists_transcripts_linked_to_patients(client, valid_patient):
    pid = client.post("/patients", json=valid_patient).json()["data"]["patient_id"]
    calls = client.app.state.call_service
    calls.start("CA-done", "3125550110")
    calls.finish("CA-done", patient_id=uuid.UUID(pid), draft_fields={}, transcript=[
        {"speaker": "agent", "text": "You're all set, Maria."},
    ])
    calls.start("CA-dropped", None)
    calls.finish("CA-dropped", patient_id=None, draft_fields={"first_name": "Jo"}, transcript=[
        {"speaker": "caller", "text": "<script>alert('x')</script>"},
    ])

    response = client.get("/calls")

    assert response.status_code == 200
    by_sid = {c["call_sid"]: c for c in response.json()["data"]}
    assert by_sid["CA-done"]["status"] == "completed"
    assert by_sid["CA-done"]["patient_id"] == pid
    assert by_sid["CA-dropped"]["status"] == "abandoned"
    # Returned as data, verbatim; the page renders it as text.
    assert by_sid["CA-dropped"]["transcript"][0]["text"] == "<script>alert('x')</script>"


def test_calls_limit_is_validated(client):
    assert client.get("/calls", params={"limit": 0}).status_code == 400
