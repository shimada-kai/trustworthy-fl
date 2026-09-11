from fastapi.testclient import TestClient

from secure_fl.tee.api import app, verified_updates


client = TestClient(app)


def setup_function():
    verified_updates.clear()


def test_submit_update_and_aggregate_round():
    response1 = client.post(
        "/submit_update",
        json={
            "round_id": 1,
            "client_id": "0",
            "state": {
                "weight": [1.0, 3.0],
            },
        },
    )

    assert response1.status_code == 200
    assert response1.json()["accepted_updates"] == 1

    response2 = client.post(
        "/submit_update",
        json={
            "round_id": 1,
            "client_id": "1",
            "state": {
                "weight": [5.0, 7.0],
            },
        },
    )

    assert response2.status_code == 200
    assert response2.json()["accepted_updates"] == 2

    assert 1 in verified_updates
    assert len(verified_updates[1]) == 2

    aggregate_response = client.post(
        "/aggregate_round",
        json={
            "round_id": 1,
        },
    )

    assert aggregate_response.status_code == 200

    result = aggregate_response.json()

    assert result["round_id"] == 1
    assert result["accepted_clients"] == 2
    assert result["aggregated_state"]["weight"] == [3.0, 5.0]
    assert result["aggregation_time_ms"] >= 0.0

    # Individual client updates must be removed after aggregation.
    assert 1 not in verified_updates

def test_duplicate_client_update_is_rejected():
    first = client.post(
        "/submit_update",
        json={
            "round_id": 1,
            "client_id": "0",
            "state": {
                "weight": [1.0, 2.0],
            },
        },
    )

    assert first.status_code == 200

    duplicate = client.post(
        "/submit_update",
        json={
            "round_id": 1,
            "client_id": "0",
            "state": {
                "weight": [9.0, 9.0],
            },
        },
    )

    assert duplicate.status_code == 409
    assert "duplicate update" in duplicate.json()["detail"]

    assert len(verified_updates[1]) == 1
