import os
import shutil
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_PATH = "chroma"
# Automatic selection of the correct folder
if os.path.exists(os.path.join("data", "knowledge_base")):
    DATA_PATH = os.path.join("data", "knowledge_base")
else:
    DATA_PATH = "data"

def main():
    if os.path.exists(CHROMA_PATH):
        print("Cleaning up the old database...")
        shutil.rmtree(CHROMA_PATH)

    print("Reading fashion files and splitting them...")
    documents = []
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data folder '{DATA_PATH}' does not exist!")
        return

    files = os.listdir(DATA_PATH)
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(DATA_PATH, file)
            with open(file_path, "r", encoding="utf-8") as f:
                full_text = f.read()
                
                # Segmenting texts based on the ID word: to ensure the extraction of the 17 shanks
                raw_chunks = full_text.split("ID:")
                for raw_chunk in raw_chunks:
                    raw_chunk = raw_chunk.strip()
                    if raw_chunk:
                        full_chunk_text = "ID: " + raw_chunk
                        if "—---" in full_chunk_text:
                            full_chunk_text = full_chunk_text.split("—---")[0].strip()
                        
                        doc = Document(
                            page_content=full_chunk_text,
                            metadata={"source": file}
                        )
                        documents.append(doc)

    print(f"Successfully extracted {len(documents)} shanks from your files!")

    if documents:
        print("Saving and embedding data in Chroma DB (this may take a moment)...")
        # Merging the embedding function directly here without an external file
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        db = Chroma.from_documents(
            documents, embeddings, persist_directory=CHROMA_PATH
        )
        print(f"Successfully indexed! The database is now saved in the folder '{CHROMA_PATH}'")
    else:
        print("No text files found in the data folder!")

if __name__ == "__main__":
    main()