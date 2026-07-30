#754b1650784a61467d23674c90d1a4ee
import os 

import requests

API_KEY = os.environ.get("WEATHER_API_KEY")

lat = 43.6532
lon = -79.3832
city = "Toronto"

url = f"https://api.openweathermap.org/data/2.5/weather?q={city},CA&appid={API_KEY}&units=metric"

# url = f"https://api.openweathermap.org/data/4.0/onecall/current?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"

response = requests.get(url)

if response.status_code == 200:
    weather_data = response.json()
    print(weather_data)

    print("\nTemperature:", weather_data["main"]["temp"], "°C")
    print("Feels like:", weather_data["main"]["feels_like"], "°C")
    print("Humidity:", weather_data["main"]["humidity"], "%")
    print("Weather:", weather_data["weather"][0]["description"])
    print("Wind:", weather_data["wind"]["speed"], "m/s")
else:
    print("Error:", response.status_code, response.text)

