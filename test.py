import requests
r = requests.post("http://127.0.0.2:8000/chat", json={"message": "hey"})
print(r.status_code)
