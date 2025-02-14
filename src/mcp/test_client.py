import json
import sys

def send_request():
    request = {
        "type": "tool",
        "name": "test_query",
        "args": {
            "query": "Testanfrage"
        }
    }
    
    # Sende die Anfrage
    json_request = json.dumps(request)
    print(json_request, flush=True)
    
    # Lese die Antwort
    response = sys.stdin.readline()
    try:
        parsed_response = json.loads(response)
        print("\nAntwort vom Server:")
        print(json.dumps(parsed_response, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print("Fehler beim Parsen der Antwort:", response)

if __name__ == "__main__":
    send_request()
