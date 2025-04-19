import requests

# Replace with your Ngrok URL
NGROK_URL = "https://5283-34-169-23-150.ngrok-free.app/generate"

# Function to send data to the API
def send_data(input_text):
    # Define the payload
    payload = {"input_text": input_text}

    # Send a POST request to the Ngrok URL
    try:
        response = requests.post(NGROK_URL, json=payload)
        response.raise_for_status()  # Raise an error if the request fails
        return response.json()  # Return the JSON response
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Test the API with an example input
    test_input = "he is a badass boy"
    print(f"Input: {test_input}")

    # Send the input text to the API
    result = send_data(test_input)

    # Print the response
    print(f"Response: {result.get('response', result)}")
