from  sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-MiniLM-L6-v2")

sentence = "Python is Programming Language."

embedding = model.encode(sentence)

print("\n\n\n")
print(embedding,"\n\n\n")
print(len(embedding),"\n\n\n")

documents = ["Python is Programming Language.",
             "AWS Lambda is Serverless",
             "Cricket is Sport"]

embedding_doc = model.encode(documents)
print(embedding_doc,"\n\n\n")
print(embedding_doc.shape,"\n\n\n")
print(embedding_doc[0],"\n\n\n")
print(embedding_doc[1],"\n\n\n")
print(embedding_doc[2],"\n\n\n")
print(len(embedding_doc),"\n\n\n")

query = "What is Serverless Computing"

embedding_query = model.encode(query)

print(embedding_query,"\n\n\n")
print(len(embedding_query),"\n\n\n")

# Compare Similarity

score = cosine_similarity([embedding_query],embedding_doc)

print(score,"\n\n\n")

##[[0.1979264  0.48737037 0.13630848]]  result will be 48 %

# Get the Best Match

best_match = score.argmax()

print(best_match,"\n\n\n")

print(documents[best_match]) ## now retreve the document

##Output : AWS Lambda is Serverless

# Congratulations! 🎉

# You just built a simple semantic search engine.