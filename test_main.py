from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "World"}


def test_predict_positive():
    response = client.post("/predict/",
                           json={"text": "I like machine learning!"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'POSITIVE'


def test_predict_negative():
    response = client.post("/predict/",
                           json={"text": "I hate machine learning!"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['label'] == 'NEGATIVE'


def test_keywords():
    text = "This is a test text. It contains some keywords\
            like Python, FastAPI, and testing."
    response = client.post("/keywords/",
                           json={"text": text})
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert all(isinstance(keyword, str) for keyword in response.json())


def test_translate():
    text = "Hello!"
    target_language = "fr"  # French
    response = client.post("/translate/",
                           json={"text": text,
                                 "target_language": target_language})
    assert response.status_code == 200
    assert isinstance(response.json(), str)
