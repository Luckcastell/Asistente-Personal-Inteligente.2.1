from sentence_transformers import SentenceTransformer

# Descarga el modelo desde HuggingFace
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Lo guarda en carpeta local
model.save("./models/all-MiniLM-L6-v2")
