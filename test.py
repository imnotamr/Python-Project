import requests

# Specify the API endpoint
url = "http://127.0.0.1:8000/predict"

# Open the image file in binary mode
img_path = "Screenshot 2024-11-01 204217.png"
with open(img_path, "rb") as image_file:
    # Send the POST request with the image file
    response = requests.post(url, files={"file": image_file})

# Print the response (predictions)
print(response.json())