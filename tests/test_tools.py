import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weather_tool import get_weather

class TestWeatherTool(unittest.TestCase):
    @patch('weather_tool.requests.get')
    def test_get_weather_success(self, mock_get):
        # Mocking environment variable
        with patch.dict(os.environ, {"OPENWEATHERMAP_API_KEY": "test_key"}):
            # Mocking response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "weather": [{"description": "clear sky"}],
                "main": {"temp": 25.0},
                "name": "Test City"
            }
            mock_get.return_value = mock_response

            result = get_weather.invoke({"city": "Test City"})
            self.assertIn("clear sky", result)
            self.assertIn("25.0°C", result)

    def test_get_weather_no_key(self):
         with patch.dict(os.environ, {}, clear=True):
             result = get_weather.invoke({"city": "London"})
             self.assertIn("API key not found", result)

if __name__ == '__main__':
    unittest.main()
