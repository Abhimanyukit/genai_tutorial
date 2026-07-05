from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

sentence_doc = ["Python is Programming language","Aws Lambda is Serverless",""
"Cricket is the sport","Footbal is sport","Delhi is capital of India","Landon is capital of England"]

embedding_doc = model.encode(sentence_doc)

print(embedding_doc,"\n\n\n")
print(embedding_doc.shape,"\n\n\n")

dimension = embedding_doc.shape[1]

print(dimension,"\n\n\n")

##Create a FAISS Index
# Think of this as creating an empty bookshelf.
index = faiss.IndexFlatL2(dimension)

print("\n\n\n index:",index)

# Add Vectors into FAISS
index.add(np.array(embedding_doc).astype("float32"))

# Why float32?

# FAISS expects vectors in the float32 data type.

print("\n\n\n index:",index)

query = "Tell me about Serveless Computing?"

# Convert Query into a Vector

query_vector = model.encode([query])

# Search

distance, indices = index.search(np.array(query_vector).astype("float32"),k=1)

print("indices:",indices,"\n\n\n")
print("distance:",distance,"\n\n\n")

result = sentence_doc[indices[0][0]]

print("result:",result,"\n\n")


# Parameter 2
# k = 1

# Means:

# Return the top 1 most similar document.

# If:

# k = 3

# FAISS returns the top three results.


query2 = "Tell me about Landon"
query2_vector = model.encode([query2])

print("query2_vector:",query2_vector,"\n\n\n")


distance, indices = index.search(np.array(query2_vector).astype("float32"),k=1)

print("distance:",distance,"\n\n\n")
print("indices:",indices,"\n\n\n")

result = sentence_doc[indices[0][0]]

print("result:",result,"\n\n\n")












