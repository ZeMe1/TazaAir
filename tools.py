import json
import os
from datetime import datetime
import requests
from langchain_core.tools import tool

DIARY_FILE = os.path.join(os.path.dirname(__file__), "diary.json")

SPOTS = [
    (["medeu", "медеу", "медео"], "Medeu (Highland)", 43.1575, 77.0589),
    (["shymbulak", "chimbulak", "шымбулак", "чимбулак"], "Shymbulak Resort", 43.1283, 77.0808),
    (["kokzhailau", "kokzhaylyau", "кокжайляу", "кокжайлау"], "Kok-Zhailau Plateau", 43.1408, 77.0056),
    (["bigalmatylake", "bao", "бао", "большоеалматинскоеозеро"], "Big Almaty Lake", 43.0506, 76.9850),
    (["burabay", "borovoe", "бурабай", "боровое"], "Burabay", 53.0839, 70.2864),
]


def find_coordinates(place: str):
    clean = place.strip().lower()
    compact = clean.replace("-", "").replace(" ", "")

    for aliases, title, lat, lon in SPOTS:
        if compact in aliases:
            return {"name": title, "country": "Kazakhstan", "lat": lat, "lon": lon}

    lang = "ru" if any("а" <= ch <= "я" or ch == "ё" for ch in clean) else "en"
    response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": place, "count": 1, "language": lang, "format": "json"},
        timeout=10,
    )
    response.raise_for_status()
    results = response.json().get("results")
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
    """Get real-time air quality (PM2.5, PM10, CO, and US AQI) for any city or mountain spot such as Almaty, Astana, Medeu, Shymbulak, Kok-Zhailau, or Big Almaty Lake."""
    try:
        location = find_coordinates(city)
        if not location:
            return f"Could not find location '{city}'. Please check the name."

        response = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": location["lat"],
                "longitude": location["lon"],
                "current": "pm2_5,pm10,carbon_monoxide,us_aqi",
            },
            timeout=10,
        )
        response.raise_for_status()
        current = response.json().get("current", {})

        return (
            f"Location: {location['name']}, {location['country']} | "
            f"PM2.5: {current.get('pm2_5')} ug/m3 | "
            f"PM10: {current.get('pm10')} ug/m3 | "
            f"CO: {current.get('carbon_monoxide')} ug/m3 | "
            f"US AQI: {current.get('us_aqi')}"
        )
    except Exception as error:
        return f"Air quality service error for '{city}': {error}"


@tool
def get_weather_and_wind(city: str) -> str:
    """Get current temperature, wind speed, humidity, and precipitation for a city or mountain spot to check if smog will disperse or stay trapped."""
    try:
        location = find_coordinates(city)
        if not location:
            return f"Could not find location '{city}'. Please check the name."

        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": location["lat"],
                "longitude": location["lon"],
                "current": "temperature_2m,wind_speed_10m,precipitation,relative_humidity_2m",
            },
            timeout=10,
        )
        response.raise_for_status()
        current = response.json().get("current", {})
        wind = current.get("wind_speed_10m")

        if wind is not None and wind < 6:
            dispersion = "Very weak wind (smog stays trapped)."
        elif wind is not None and wind < 15:
            dispersion = "Moderate wind (partial dispersion)."
        else:
            dispersion = "Strong wind (blows smog away)."

        return (
            f"Location: {location['name']} | "
            f"Temperature: {current.get('temperature_2m')} C | "
            f"Wind: {wind} km/h | "
            f"Precipitation: {current.get('precipitation')} mm | "
            f"Humidity: {current.get('relative_humidity_2m')}% | "
            f"Dispersion: {dispersion}"
        )
    except Exception as error:
        return f"Weather service error for '{city}': {error}"


@tool
def calculate_cigarettes(pm25: float, hours_outside: float = 24.0) -> str:
    """Calculate how many cigarettes breathing outdoor air equals based on PM2.5 (ug/m3) and hours spent outside, plus window and FFP2 mask advice."""
    try:
        if pm25 < 0 or hours_outside < 0:
            return "PM2.5 and hours_outside must be non-negative."

        cigs_per_day = round(pm25 / 22.0, 2)
        cigs_for_exposure = round((pm25 / 22.0) * (hours_outside / 24.0), 2)

        if pm25 <= 12:
            advice = "Clean air. Safe to open windows and exercise outside."
        elif pm25 <= 35:
            advice = "Moderate air. Brief ventilation is fine; sensitive groups should limit heavy cardio."
        elif pm25 <= 55:
            advice = "Polluted air. Keep windows closed, use an air purifier, and shorten walks."
        else:
            advice = "Heavy smog! Keep windows shut, run an air purifier, and wear an FFP2/N95 mask outside."

        return (
            f"PM2.5: {pm25} ug/m3 | Exposure: {hours_outside}h | "
            f"Inhaled during exposure: {cigs_for_exposure} cigarettes | "
            f"24h equivalent: {cigs_per_day} cigarettes/day | "
            f"Advice: {advice}"
        )
    except Exception as error:
        return f"Calculation error: {error}"


@tool
def air_diary(action: str, city: str = "", note: str = "") -> str:
    """Save or read personal air quality notes in diary.json. Use action='save' (with city and note) or action='read'."""
    try:
        mode = action.strip().lower()
        entries = []

        if os.path.exists(DIARY_FILE) and os.path.getsize(DIARY_FILE) > 0:
            with open(DIARY_FILE, "r", encoding="utf-8") as file:
                entries = json.load(file)

        if mode == "read":
            if not entries:
                return "Air diary is empty."
            return "\n".join(f"[{item['timestamp']}] {item['city']}: {item['note']}" for item in entries)

        if mode == "save":
            if not city or not note:
                return "Provide both 'city' and 'note' to save an entry."
            entries.append(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "city": city,
                    "note": note,
                }
            )
            with open(DIARY_FILE, "w", encoding="utf-8") as file:
                json.dump(entries, file, ensure_ascii=False, indent=2)
            return f"Saved to diary: {city} — {note}"

        return "Unknown action. Use 'save' or 'read'."
    except Exception as error:
        return f"Diary error: {error}"
