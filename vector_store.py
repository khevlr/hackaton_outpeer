import os
import json
import torch
import numpy as np
from typing import List, Dict, Any
import chromadb
from tqdm import tqdm
from resume_parser_cop import roberta_similarity
import hashlib
import pickle
import time

# Path settings
VACANCY_DIR = 'vacancies'
CHROMA_DB_PATH = 'chroma_db'
CACHE_FILE = 'vacancy_cache.pkl'

def get_directory_hash() -> str:
    """Create a hash of all files in the vacancy directory to detect changes."""
    hash_str = ""
    
    # Get list of all files and their modification times
    files_info = []
    for filename in sorted(os.listdir(VACANCY_DIR)):
        if filename.endswith('.json'):
            file_path = os.path.join(VACANCY_DIR, filename)
            mod_time = os.path.getmtime(file_path)
            file_size = os.path.getsize(file_path)
            files_info.append(f"{filename}:{mod_time}:{file_size}")
    
    # Create a hash of all filenames and their info
    hash_str = "\n".join(files_info)
    return hashlib.md5(hash_str.encode()).hexdigest()

def should_rebuild_db() -> bool:
    """Check if the database should be rebuilt based on changes in the vacancy directory."""
    current_hash = get_directory_hash()
    
    # Try to load the previous hash
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'rb') as f:
                cache_data = pickle.load(f)
                previous_hash = cache_data.get('hash', '')
                
                # If hashes match, no need to rebuild
                if previous_hash == current_hash:
                    return False
    except Exception:
        pass  # If anything goes wrong, rebuild to be safe
    
    # Save the current hash
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump({'hash': current_hash, 'timestamp': time.time()}, f)
    
    return True

def load_vacancies() -> List[Dict[str, Any]]:
    """Load all vacancy JSON files and return them as a list of dictionaries."""
    vacancies = []
    
    for filename in tqdm(os.listdir(VACANCY_DIR), desc="Loading vacancies"):
        if filename.endswith('.json'):
            file_path = os.path.join(VACANCY_DIR, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    vacancy_data = json.load(f)
                    
                # Extract company and position from filename
                parts = filename.replace('.json', '').split('_')
                if len(parts) > 1:
                    company = parts[0]
                    position = ' '.join(parts[1:])
                else:
                    company = ""
                    position = filename.replace('.json', '')
                
                # Add metadata
                vacancy_data['filename'] = filename
                if 'company_name' not in vacancy_data:
                    vacancy_data['company_name'] = company
                if 'position_name' not in vacancy_data:
                    vacancy_data['position_name'] = position
                
                vacancies.append(vacancy_data)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return vacancies

def create_embeddings(texts: List[str]) -> List[List[float]]:
    """Create embeddings for a list of texts using the roberta model."""
    embeddings = []
    
    for text in tqdm(texts, desc="Creating embeddings"):
        embedding = roberta_similarity(text).numpy().tolist()
        embeddings.append(embedding)
    
    return embeddings

def build_vector_db(force_rebuild: bool = False) -> chromadb.Collection:
    """
    Build or load a Chroma vector database with vacancy embeddings.
    
    Args:
        force_rebuild: If True, rebuild the database even if it exists.
        
    Returns:
        ChromaDB collection object
    """
    # Check if we need to rebuild based on vacancies folder changes
    needs_rebuild = force_rebuild or should_rebuild_db()
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Check if collection exists
    try:
        collection = chroma_client.get_collection("vacancies")
        if needs_rebuild:
            chroma_client.delete_collection("vacancies")
            raise ValueError("Rebuilding database")
        print(f"Loaded existing collection with {collection.count()} documents")
        return collection
    except (ValueError, Exception) as e:
        # Create new collection if it doesn't exist or we're forcing rebuild
        print(f"Creating new collection: {str(e)}")
        collection = chroma_client.create_collection(
            name="vacancies"
        )
    
    # Load vacancies
    vacancies = load_vacancies()
    
    if not vacancies:
        print("No vacancies found")
        return collection
    
    # Prepare data for Chroma
    ids = [f"vacancy_{i}" for i in range(len(vacancies))]
    texts = [v.get('summary', '') for v in vacancies]
    
    # Create embeddings
    print("Creating embeddings...")
    embeddings = create_embeddings(texts)
    
    # Prepare metadata
    metadatas = []
    for vacancy in vacancies:
        metadata = {
            'filename': vacancy.get('filename', ''),
            'company_name': vacancy.get('company_name', ''),
            'position_name': vacancy.get('position_name', ''),
            'skills': vacancy.get('skills', '')
        }
        metadatas.append(metadata)
    
    # Add data to collection in batches (to avoid memory issues)
    batch_size = 100
    for i in tqdm(range(0, len(vacancies), batch_size), desc="Adding to Chroma"):
        end_idx = min(i + batch_size, len(vacancies))
        collection.add(
            ids=ids[i:end_idx],
            embeddings=embeddings[i:end_idx],
            documents=[json.dumps(v) for v in vacancies[i:end_idx]],
            metadatas=metadatas[i:end_idx]
        )
    
    print(f"Built vector database with {collection.count()} vacancies")
    return collection

def search_similar_vacancies(query_embedding, n_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search for vacancies similar to the query embedding.
    
    Args:
        query_embedding: The embedding vector of the query
        n_results: Number of results to return
        
    Returns:
        List of vacancy dictionaries with similarity scores
    """
    # Check if database needs to be rebuilt first
    if should_rebuild_db():
        print("Vacancies have changed. Rebuilding database...")
        build_vector_db(force_rebuild=True)
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    try:
        collection = chroma_client.get_collection("vacancies")
    except Exception:
        print("Vector database not found. Building...")
        collection = build_vector_db()
    
    # Convert embedding to list if it's a tensor
    if isinstance(query_embedding, torch.Tensor):
        query_embedding = query_embedding.numpy().tolist()
    
    # Search the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    # Process results
    vacancies = []
    for i, doc in enumerate(results['documents'][0]):
        vacancy_data = json.loads(doc)
        # Calculate similarity score (1 - distance) for easy understanding
        similarity_score = 1 - results['distances'][0][i]
        vacancy_data['similarity_score'] = similarity_score
        vacancy_data['metadata'] = results['metadatas'][0][i]
        vacancies.append(vacancy_data)
    
    return vacancies

if __name__ == "__main__":
    # Build the vector database if run directly
    collection = build_vector_db(force_rebuild=True)
    print(f"Created vector database with {collection.count()} documents") 