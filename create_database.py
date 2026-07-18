import os
import shutil
from langchain_chroma import Chroma
from langchain_core.documents import Document
from custom_embeddings import CustomEmbeddings

CHROMA_PATH = "chroma"
# Automatic selection of the correct folder
if os.path.exists(os.path.join("data", "knowledge_base")):
    DATA_PATH = os.path.join("data", "knowledge_base")
else:
    DATA_PATH = "data"

def main():
    #1. Clean up the old database, if it exists, to prevent data overlap.
    if os.path.exists(CHROMA_PATH):
        print("Cleaning up the old database...")
        shutil.rmtree(CHROMA_PATH)
    # 2. Read the files and extract the Chunks based on ID:
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
                
                # splitting based on the word "ID:" to ensure extraction of the 17 chunks
                raw_chunks = full_text.split("ID:")
                for raw_chunk in raw_chunks:
                    raw_chunk = raw_chunk.strip()
                    if raw_chunk:
                        full_chunk_text = "ID: " + raw_chunk
                        if "—---" in full_chunk_text:
                            full_chunk_text = full_chunk_text.split("—---")[0].strip()
                        # Convert the chunk to a LangChain document with source metadata
                        doc = Document(
                            page_content=full_chunk_text,
                            metadata={"source": file}
                        )
                        documents.append(doc)

    print(f"Successfully extracted {len(documents)} shanks from your files!")
    # 3. Save the chunks in a local Chroma DB
    if documents:
        print("Saving and embedding data in Chroma DB (this may take a moment)...")
        embeddings = CustomEmbeddings()
        
        db = Chroma.from_documents(
            documents, embeddings, persist_directory=CHROMA_PATH
        )
        
        print(f"Successfully indexed! The database is now saved in the folder '{CHROMA_PATH}'")
    else:
        print("No text files found in the data folder!")

if __name__ == "__main__":
    main()
