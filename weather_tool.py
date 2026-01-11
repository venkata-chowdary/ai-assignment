import os
import requests
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Fetch the current weather for a given city using OpenWeatherMap API."""
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return "Error: OpenWeatherMap API key not found."

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        weather_desc = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        city_name = data["name"]
        
        return f"The current weather in {city_name} is {weather_desc} with a temperature of {temp}°C."
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except Exception as err:
        return f"An error occurred: {err}"
