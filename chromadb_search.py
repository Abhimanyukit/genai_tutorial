import os 
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
folder = "policies"

# Read all policy File 

for filename in os.listdir(folder):
    print("filename:",filename,"\n\n\n")
    with open(os.path.join(folder,filename),"r") as f:
        documents.append(f.read())

print("documents:",documents,"\n\n\n")


embedding_policy = model.encode(documents).tolist()

# Create CromaDb

client = chromadb.PersistentClient(path = "./chromadb")

# create Collection 

collection = client.get_or_create_collection(name = "employee_policies")

## Store Document

collection.add(
ids = ["1","2","3"],
documents=documents,
embeddings=embedding_policy,
metadatas=[{filename:"hr_policy"},
           {filename:"leave_policy"},
           {filename:"travel_policy"}]
)

print("collection:",collection,"\n\n\n")

query = "How many casual leaves are allowed?"

query_embedding = model.encode(query).tolist()

# Search ChromaDb
#ChromaDB does NOT answer questions. It only retrieves relevant documents.

#The LLM (GPT, Claude, Gemini, Llama, etc.) reads those retrieved documents and generates the final answer.
result = collection.query(
    query_embeddings= [query_embedding],
    n_results= 2
)

print("result:",result,"\n\n\n")

# Send Context to LLM 

context =result["documents"][0][0]

print("context:",context,"\n\n\n")

question = "How many casual leaves are allowed?"

prompt = f"""

    Answer using context below.
    Context:{context}

    Question:{question}

"""

for i, doc in enumerate(result["documents"][0]):
    print(f"Result {i+1}:")
    print(doc)
    print("-" * 50)



