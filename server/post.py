import requests
import json
import numpy as np
import random
import random

new_data_sequence = [
    # Data entry 1
    [[random.random() for _ in range(16)]],  # Timestep 1
    # Data entry 2
    [[random.random() for _ in range(16)]],  # Timestep 1
    # Data entry 3
    [[random.random() for _ in range(16)]],  # Timestep 1
    # Data entry 2
    [[random.random() for _ in range(16)]],  # Timestep 1
    # Data entry 2
    [[random.random() for _ in range(16)]],  # Timestep 1
    # Data entry 3
    [[random.random() for _ in range(16)]],  # Timestep 1
    # Data entry 3
    [[random.random() for _ in range(16)]],  # Timestep 1
    # Data entry 2
    [[random.random() for _ in range(16)]],  # Timestep 1
    # Data entry 3
    [[random.random() for _ in range(16)]],  # Timestep 1
    # Data entry 2
    [[random.random() for _ in range(16)]],  # Timestep 1
    # Data entry 2
    [[random.random() for _ in range(16)]],  # Timestep 1
    # Data entry 3
    [[random.random() for _ in range(16)]],  # Timestep 1
    # Data entry 13
    [[random.random() for _ in range(16)]],  # Timestep 1
    # Data entry 2
    [[random.random() for _ in range(16)]],  # Timestep 1
    # Data entry 3
    [[random.random() for _ in range(16)]],  # Timestep 1
    # Data entry 2
    [[random.random() for _ in range(16)]],  # Timestep 1
    # Data entry 2
    [[random.random() for _ in range(16)]],  # Timestep 1
    # Data entry 3
    [[random.random() for _ in range(16)]],  # Timestep 1
    # Data entry 19
    [[random.random() for _ in range(16)]],  # Timestep 1
    # Data entry 2
    [[random.random() for _ in range(16)]],  # Timestep 1
    # Data entry 3
    [[random.random() for _ in range(16)]],  # Timestep 1
    # Data entry 2
    [[random.random() for _ in range(16)]],  # Timestep 1
    # Data entry 2
    [[random.random() for _ in range(16)]],  # Timestep 1
    # Data entry 3
    [[random.random() for _ in range(16)]],  # Timestep 1
]
# for shape
input_data = np.array(new_data_sequence)

# Print the shape of the input data being posted
print("Shape of input data being posted:", input_data.shape)

url = 'https://little-snow-18911.pktriot.net/predict'
data = {'input': input_data.tolist()}  # Convert numpy array to list for JSON serialization 
headers = {'Content-Type': 'application/json'}

response = requests.post(url, data=json.dumps(data), headers=headers)

# Check if the request was successful 
if response.status_code == 200:
    try:
        # Try to decode the response as JSON
        prediction = response.json()
        print("Prediction:", prediction)
    except json.decoder.JSONDecodeError:
        # Handle JSONDecodeError
        print("Error: Unable to decode JSON response")
else:
    # Handle unsuccessful request
    print("Error:", response.text)
