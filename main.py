import speech_recognition as sr
import os
import pyttsx3
import webbrowser
import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import requests
import wikipedia
import time

# Initialize DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize conversation history
chat_history_ids = None

# API keys
weather_api_key = "7a81b3fa5f2bbba4049415903833cc06"  # Replace with your actual API key
news_api_key = "f3b4e5a4659e4890a57ae19766a19fb3"  # Replace with your actual API key

def say(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        try:
            print("Recognizing...")
            query = r.recognize_google(audio, language="en-in")
            print(f"User said: {query}")
            return query
        except Exception as e:
            print(e)
            return "Some Error Occurred. Sorry from Maverick A.I"

def play_music(file_path):
    try:
        os.startfile(file_path)
    except Exception as e:
        print(e)
        say("Unable to play the music file.")

def chat_mode():
    global chat_history_ids
    print("Entering chat mode. Type 'exit chat' to leave chat mode.")
    say("Entering chat mode. Type 'exit chat' to leave chat mode.")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == "exit chat":
            print("Exiting chat mode.")
            say("Exiting chat mode.")
            break
        
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids
        
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            attention_mask=torch.ones(bot_input_ids.shape, dtype=torch.long)
        )
        
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"Maverick: {response}")
        say(response)

def get_weather(city_name, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    complete_url = f"{base_url}?q={city_name}&appid={api_key}&units=metric"
    response = requests.get(complete_url)
    data = response.json()
    
    print("API Response:", data)  # Debugging line
    
    if response.status_code == 200 and "main" in data:
        main = data["main"]
        weather = data["weather"][0]
        temperature = main["temp"]
        description = weather["description"]
        return f"The current temperature in {city_name} is {temperature}°C with {description}."
    else:
        return "City not found or an error occurred while fetching the weather data."

def get_news(api_key):
    base_url = "https://newsapi.org/v2/top-headlines"
    country = "us"  # You can change this to your desired country
    complete_url = f"{base_url}?country={country}&apiKey={api_key}"
    response = requests.get(complete_url)
    data = response.json()
    
    print("API Response:", data)  # Debugging line
    
    if data["status"] == "ok":
        articles = data["articles"][:5]  # Get top 5 news articles
        news = []
        for article in articles:
            news.append(article["title"])
        return news
    else:
        return "Unable to fetch news."

def get_joke():
    joke_api_url = "https://v2.jokeapi.dev/joke/Any"
    response = requests.get(joke_api_url)
    data = response.json()
    if data["type"] == "single":
        return data["joke"]
    elif data["type"] == "twopart":
        return f"{data['setup']} ... {data['delivery']}"
    else:
        return "Couldn't fetch a joke at the moment."

def search_wikipedia(query):
    try:
        results = wikipedia.summary(query, sentences=2)
        return results
    except Exception as e:
        return "Error fetching data from Wikipedia."

def set_timer(duration):
    say(f"Timer set for {duration} seconds.")
    time.sleep(duration)
    say("Time's up!")

if __name__ == '__main__':
    print('Welcome to Maverick A.I')
    say("Hello, I am Maverick A.I")

    # Ask for input mode only once
    mode = input("Choose input mode (voice/text): ").strip().lower()

    while mode not in ["voice", "text"]:
        say("Invalid mode. Please choose either 'voice' or 'text'.")
        mode = input("Choose input mode (voice/text): ").strip().lower()

    while True:
        if mode == "voice":
            print("Listening...")
            query = takeCommand()
        elif mode == "text":
            query = input("Enter your command: ").strip().lower()

        # Define sites to open
        sites = [["youtube", "https://www.youtube.com"], ["wikipedia", "https://www.wikipedia.com"], ["google", "https://www.google.com"]]

        for site in sites:
            if f"open {site[0]}".lower() in query.lower():
                say(f"Opening {site[0]} master !")
                webbrowser.open(site[1])

        if "open music" in query:
            musicPath = r"C:\Users\Chander\Desktop\Maverick\audio\un_poco_loco.mp3"  # Use raw string
            play_music(musicPath)

        elif "the time" in query:
            hour = datetime.datetime.now().strftime("%H")
            minute = datetime.datetime.now().strftime("%M")
            say(f"Sir, the time is {hour} hours and {minute} minutes")

        elif "chat with me" in query.lower():
            chat_mode()

        elif "weather" in query:
            city = query.split("weather in ")[1].strip()
            weather_info = get_weather(city, weather_api_key)
            print(weather_info)
            say(weather_info)

        elif "news" in query:
            news = get_news(news_api_key)
            if news:
                for article in news:
                    print(article)
                    say(article)
            else:
                print("No news available")
                say("No news available")

        elif "joke" in query:
            joke = get_joke()
            print(joke)
            say(joke)

        elif "wikipedia" in query:
            search_term = query.replace("wikipedia", "").strip()
            result = search_wikipedia(search_term)
            print(result)
            say(result)

        elif "set timer" in query:
            duration = int(query.split("set timer for ")[1].split(" seconds")[0].strip())
           

