import os
import requests
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """
    fetches current weather for a given city via openweathermap api.
    """
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    
    # checking if api key is there or not
    if not api_key:
        return "Error: API key not found. please check .env file."

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric" # getting temp in celsius
    }

    try:
        # calling the api
        response = requests.get(base_url, params=params)
        data = response.json()

        if response.status_code == 200:
            # parsing the main details
            weather_desc = data["weather"][0]["description"]
            temp = data["main"]["temp"]
            feels_like = data["main"]["feels_like"]
            return f"Weather in {city}: {weather_desc}, Temp: {temp}°C, Feels like: {feels_like}°C"
        else:
            return f"Error getting weather: {data.get('message', 'Unknown error')}"

    except Exception as e:
        return f"failed to connect to weather api: {e}"
