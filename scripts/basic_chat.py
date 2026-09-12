import requests

while True:
    message = input("You: ")
    r = requests.post("http://127.0.0.2:8000/chat", json={"message": message})
    if r.status_code != 200:
        print("Error:", r.status_code)
        break

    print("Amadeus:", r.json().get("response", ""))