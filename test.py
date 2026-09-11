import requests
r = requests.post("http://127.0.0.1:8000/chat", json={"session_id": "test", "message": "hey","timestamp": "2024-06-05T12:00:00Z"})
print(r.json())