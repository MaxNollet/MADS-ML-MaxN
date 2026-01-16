import pytest
import requests


@pytest.mark.integration
class TestDocker:
    def setup_method(self):
        self.base_url = "http://localhost:8000"

    def test_health(self):
        response = requests.get(f"{self.base_url}/health")
        assert response.json()["status"] == "healthy"

    def test_add(self):
        response = requests.get(f"{self.base_url}/add/2/3")
        assert response.status_code == 200
        assert response.json()["result"] == 5

    def test_division(self):
        # Test division
        response = requests.get(f"{self.base_url}/divide/6/2")
        assert response.status_code == 200
        assert response.json()["result"] == 3
