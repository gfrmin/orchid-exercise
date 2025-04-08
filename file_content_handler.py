from pathlib import Path
import faiss
import numpy as np
from tqdm import tqdm

from codebase_reader import REPO_PATH
from llm_provider import llm_provider, LLMProvider


class FileContentIndex:
    def __init__(self, llm_provider_: LLMProvider):
        self.llm_provider = llm_provider_
        # Get embedding dimension from a test embedding
        test_embedding = self.llm_provider.client.embeddings.create(
            model=self.llm_provider.embedding_model_name,
            input="test"
        )
        self.embedding_dimension = len(test_embedding.data[0].embedding)
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        self.file_paths: list[str] = []
        self.text_contents: list[str] = []
        
    def process_directory(self, directory_path: str) -> None:
        """Recursively process all files in the directory and create embeddings."""
        directory = Path(directory_path)
        
        # Get all files recursively, excluding directories
        files = [f for f in directory.rglob("*") if f.is_file()]
        
        print(f"Found {len(files)} files. Processing...")
        
        for file_path in tqdm(files, desc="Processing files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Create embedding for the content
                embedding = self.llm_provider.client.embeddings.create(
                    model=self.llm_provider.embedding_model_name,
                    input=content
                )
                
                # Convert embedding to numpy array
                embedding_array = np.array(embedding.data[0].embedding, dtype=np.float32)
                
                # Add to FAISS index
                self.index.add(np.array([embedding_array]))
                
                # Store metadata
                self.file_paths.append(str(file_path))
                self.text_contents.append(content)
                
            except (UnicodeDecodeError, IOError) as e:
                continue
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Search for similar content using a query string."""
        # Create embedding for the query
        query_embedding = self.llm_provider.client.embeddings.create(
            model=self.llm_provider.embedding_model_name,
            input=query
        )
        
        # Convert query embedding to numpy array
        query_array = np.array(query_embedding.data[0].embedding, dtype=np.float32)
        
        # Search in FAISS index
        distances, indices = self.index.search(np.array([query_array]), k)
        
        # Return results with file paths and distances
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.file_paths):  # Ensure valid index
                results.append((self.file_paths[idx], float(distance)))
        
        return results
    
    def get_content_by_path(self, file_path: str) -> str:
        """Retrieve the content of a specific file."""
        try:
            idx = self.file_paths.index(file_path)
            return self.text_contents[idx]
        except ValueError:
            return "File not found in the embedded dictionary."

file_content_handler = FileContentIndex(llm_provider)
file_content_handler.process_directory(REPO_PATH)