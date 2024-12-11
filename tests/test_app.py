import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Starbucks Wait Time Predictor" in response.data

def test_predict(client):
    response = client.post('/predict', data={'day': 'Monday', 'time': '10:30'})
    assert response.status_code == 200
    assert b"Predicted Wait Time" in response.data
