import requests
import json

# Disable Streaming

payload = {

    "model" : "llama3.2",
    "stream" : False,
    "messages":[{
        "role":"user",
        "content":"What is Python"
}
    ]
}

print("payload:",payload,"\n\n\n")
response = requests.post("http://localhost:11434/api/chat",json=payload)

print(response,"\n\n\n")

# Python expects one JSON document, but it sees multiple JSON objects, which causes:
# JSONDecodeError: Extra data

print(response.json(),"\n\n\n")
print(response.json()["message"]["content"],"\n\n\n")


# Handle Streaming Responses

payload = {

    "model" : "llama3.2",
    "messages":[{
        "role":"user",
        "content":"What is Python"
}
    ]
}

print("payload:",payload,"\n\n\n")
response = requests.post("http://localhost:11434/api/chat",json=payload, stream=True)

print(response,"\n\n\n")

output = []

for line in response.iter_lines():
    if line:
        #print("line:",line,"\n\n\n")
        data = json.loads(line)
        #print("data:",data,"\n\n\n")
        output.append(data["message"]["content"])


result = " ".join(output)
print("result:",result,"\n\n\n")
