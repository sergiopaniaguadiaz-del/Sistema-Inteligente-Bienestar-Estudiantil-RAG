from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import os

# ==================================================
# CARGAR DOCUMENTOS
# ==================================================

docs = []

folder_path = "data_rag"

for file in os.listdir(folder_path):

    if file.endswith(".txt"):

        loader = TextLoader(
            os.path.join(folder_path, file),
            encoding="utf-8"
        )

        docs.extend(loader.load())

# ==================================================
# DIVIDIR TEXTO
# ==================================================

text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

documents = text_splitter.split_documents(docs)

# ==================================================
# EMBEDDINGS
# ==================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==================================================
# CREAR BASE VECTORIAL
# ==================================================

db = FAISS.from_documents(
    documents,
    embeddings
)

# ==================================================
# GUARDAR
# ==================================================

db.save_local("vector_db")

print("Base vectorial creada correctamente")