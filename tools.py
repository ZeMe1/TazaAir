import json
import os
from datetime import datetime
import requests
from langchain_core.tools import tool

DIARY_FILE = os.path.join(os.path.dirname(__file__), "diary.json")

LOCAL_SPOTS = {
    "medeu": {"name": "Medeu (Highland)", "country": "Kazakhstan", "lat": 43.1575, "lon": 77.0589},
    "shymbulak": {"name": "Shymbulak (Mountain Resort)", "country": "Kazakhstan", "lat": 43.1283, "lon": 77.0808},
    "kok-zhailau": {"name": "Kok-Zhailau Plateau", "country": "Kazakhstan", "lat": 43.1408, "lon": 77.0056},
    "big almaty lake": {"name": "Big Almaty Lake", "country": "Kazakhstan", "lat": 43.0506, "lon": 76.9850},
    "burabay": {"name": "Burabay", "country": "Kazakhstan", "lat": 53.0839, "lon": 70.2864},
}


def find_coordinates(place: str):
    key = place.strip().lower()
    if key in LOCAL_SPOTS:
        return LOCAL_SPOTS[key]

    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    response = requests.get(
        geo_url,
        params={"name": place, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    results = data.get("results")
    if not results:
        return None

    first = results[0]
    return {
        "name": first.get("name", place),
        "country": first.get("country", "Unknown"),
        "lat": first["latitude"],
        "lon": first["longitude"],
    }


@tool
def get_air_quality(city: str) -> str:
    """Get real-time air quality data (PM2.5, PM10, CO, and US AQI) for a city or mountain spot like Almaty, Astana, Medeu, or Shymbulak."""
    try:
        location = find_coordinates(city)
        if not location:
            return f"Could not find location '{city}'. Please check the city name."

        aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": location["lat"],
            "longitude": location["lon"],
            "current": "pm2_5,pm10,carbon_monoxide,us_aqi,european_aqi",
        }
        response = requests.get(aq_url, params=params, timeout=10)
        response.raise_for_status()
        current = response.json().get("current", {})

        pm25 = current.get("pm2_5")
        pm10 = current.get("pm10")
        co = current.get("carbon_monoxide")
        us_aqi = current.get("us_aqi")

        return (
            f"Location: {location['name']}, {location['country']} | "
            f"PM2.5: {pm25} ug/m3 | PM10: {pm10} ug/m3 | "
            f"CO: {co} ug/m3 | US AQI: {us_aqi}"
        )
    except Exception as error:
        return f"Air quality service error for '{city}': {error}"


@tool
def get_weather_and_wind(city: str) -> str:
    """Get current temperature, wind speed, humidity, and precipitation for a city to check if smog will disperse or stagnate."""
    try:
        location = find_coordinates(city)
        if not location:
            return f"Could not find location '{city}'. Please check the city name."

        weather_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": location["lat"],
            "longitude": location["lon"],
            "current": "temperature_2m,wind_speed_10m,precipitation,relative_humidity_2m",
        }
        response = requests.get(weather_url, params=params, timeout=10)
        response.raise_for_status()
        current = response.json().get("current", {})

        temp = current.get("temperature_2m")
        wind = current.get("wind_speed_10m")
        rain = current.get("precipitation")
        humidity = current.get("relative_humidity_2m")

        if wind is not None and wind < 6:
            dispersion = "Very weak wind (smog tends to accumulate and stay trapped)."
        elif wind is not None and wind < 15:
            dispersion = "Moderate wind (partial air movement)."
        else:
            dispersion = "Strong wind (helps blow smog away)."

        return (
            f"Location: {location['name']} | Temperature: {temp} C | "
            f"Wind: {wind} km/h | Precipitation: {rain} mm | "
            f"Humidity: {humidity}% | Smog Dispersion: {dispersion}"
        )
    except Exception as error:
        return f"Weather service error for '{city}': {error}"


@tool
def calculate_cigarettes(pm25: float, hours_outside: float = 24.0) -> str:
    """Calculate how many cigarettes breathing outdoor air equals based on PM2.5 level (ug/m3) and hours spent outside, plus ventilation and mask advice."""
    try:
        if pm25 < 0 or hours_outside < 0:
            return "PM2.5 and hours_outside must be non-negative numbers."

        cigs_per_day = round(pm25 / 22.0, 2)
        cigs_for_exposure = round((pm25 / 22.0) * (hours_outside / 24.0), 2)

        if pm25 <= 12:
            advice = "Clean air. Safe to open windows and do outdoor sports."
        elif pm25 <= 35:
            advice = "Moderate air. Brief window ventilation is fine; sensitive groups should limit intense outdoor cardio."
        elif pm25 <= 55:
            advice = "Polluted air. Keep windows closed, use an air purifier, and shorten outdoor walks."
        else:
            advice = "Heavy smog alert! Do NOT open windows, run an air purifier indoors, and wear an FFP2/N95 mask outside."

        return (
            f"PM2.5: {pm25} ug/m3 | Exposure: {hours_outside} hours | "
            f"Cigarettes inhaled during exposure: {cigs_for_exposure} cigarettes | "
            f"Full 24-hour equivalent: {cigs_per_day} cigarettes/day | "
            f"Health Advice: {advice}"
        )
    except Exception as error:
        return f"Calculation error: {error}"


@tool
def air_diary(action: str, city: str = "", note: str = "") -> str:
    """Save or read personal air quality notes in a local JSON diary. Set action to 'save' (with city and note) or 'read'."""
    try:
        action_clean = action.strip().lower()
        entries = []

        if os.path.exists(DIARY_FILE):
            with open(DIARY_FILE, "r", encoding="utf-8") as file:
                entries = json.load(file)

        if action_clean == "read":
            if not entries:
                return "Air diary is currently empty."
            lines = [
                f"[{item['timestamp']}] {item['city']}: {item['note']}"
                for item in entries
            ]
            return "\n".join(lines)

        if action_clean == "save":
            if not city or not note:
                return "Please provide both 'city' and 'note' when saving to the diary."
            new_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "city": city,
                "note": note,
            }
            entries.append(new_entry)
            with open(DIARY_FILE, "w", encoding="utf-8") as file:
                json.dump(entries, file, ensure_ascii=False, indent=2)
            return f"Saved to diary: {city} — {note}"

        return "Unknown action. Use 'save' or 'read'."
    except Exception as error:
        return f"Diary file error: {error}"
