import requests
import json

def test_status():
    url = "http://127.0.0.1:8080/status"
    
    print("\nTesting /status endpoint:")
    print(f"GET {url}")
    
    response = requests.get(url)
    
    print("\nResponse:")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("Server Info:")
        data = response.json()
        print(json.dumps(data, indent=2))
    else:
        print(f"Error: {response.text}")

def test_inference():
    url = "http://127.0.0.1:8080/inference"
    text = "good"
    
    print("\nTesting /inference endpoint:")
    print(f"POST {url}")
    print(f"Body: {text}")
    
    headers = {
        "Content-Type": "text/plain",
        "Content-Length": str(len(text))
    }
    
    response = requests.post(url, data=text, headers=headers)
    
    print("\nResponse:")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data, indent=2))
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    try:
        test_status()
        test_inference()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Make sure the server is running on http://127.0.0.1:8080")