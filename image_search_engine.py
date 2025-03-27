import torch
import requests
import chromadb
from chromadb.config import Settings
from fastapi import HTTPException
from tenacity import retry, stop_after_attempt, wait_exponential


class ImageSearchEngine:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def __init__(self,
                 db_name="image_search_db",
                 api_url="http://localhost:8000",
                 chroma_host="localhost",
                 chroma_port=9000):
        """
        Initialize the ImageSearchEngine with ChromaDB client/server mode.

        Args:
            db_name (str): Name of the Chroma database collection.
            api_url (str): URL of the FastAPI server for embeddings.
            chroma_host (str): ChromaDB server host.
            chroma_port (int): ChromaDB server port.
        """
        # Create ChromaDB HTTP client
        self.client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port,
            settings=Settings(
                anonymized_telemetry=False
            )
        )

        # Verify connection
        self.client.heartbeat()

        # API for embeddings
        self.api_url = api_url

        # Get or create collection with specific metadata
        self.vector_db = self.client.get_or_create_collection(
            name=db_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Load existing image hashes
        self.image_hashes = self._load_image_hashes_from_db()

    def get_embedding_from_api(self, query_data: dict, endpoint: str) -> torch.Tensor:
        """
        Send a request to the FastAPI server to get embeddings for text or image.
        """
        try:
            response = requests.post(f"{self.api_url}{endpoint}", json=query_data)
            response.raise_for_status()
            embedding = response.json().get('embedding')
            return torch.tensor(embedding)
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Error communicating with embedding server: {str(e)}")

    def index_image(self, image_url: str, metadata: dict = None):
        """
        Index an image using its URL with improved error handling.
        """
        image_hash = hash(image_url)

        # Check if image already exists
        if image_hash in self.image_hashes:
            raise HTTPException(status_code=400, detail="Image URL already indexed.")

        try:
            # Get image embedding
            image_embedding = self.get_embedding_from_api({"image_url": image_url}, "/embed_image")

            # Prepare metadata
            if metadata is None:
                metadata = {}
            metadata["url"] = image_url
            metadata["image_hash"] = image_hash

            # Add to vector database
            self.vector_db.add(
                embeddings=image_embedding.numpy().tolist(),
                metadatas=[metadata],
                ids=[str(image_hash)],
            )

            # Update local hash tracking
            self.image_hashes.add(image_hash)

        except Exception as e:
            # Comprehensive error handling
            raise HTTPException(status_code=500, detail=f"Failed to index image: {str(e)}")

    def remove_image_from_index(self, image_url: str):
        """
        Remove an image from the index using its URL.
        """
        image_hash = hash(image_url)

        # Check if image exists
        if image_hash not in self.image_hashes:
            raise HTTPException(status_code=404, detail="Image URL not found in index.")

        try:
            # Remove from vector database
            self.vector_db.delete(ids=[str(image_hash)])

            # Update local hash tracking
            self.image_hashes.remove(image_hash)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to remove image: {str(e)}")

    def search_image(self, query_image_url: str, top_k: int = 5):
        """
        Search for visually similar images with improved error handling.
        """
        try:
            # Get query image embedding
            query_embedding = self.get_embedding_from_api(
                {"image_url": query_image_url},
                "/embed_image"
            )

            # Perform vector search
            results = self.vector_db.query(
                query_embeddings=query_embedding.numpy().tolist(),
                n_results=top_k
            )

            # Extract URLs and distances
            urls = [result.get('url', 'Unknown') for result in results['metadatas'][0]]
            distances = results['distances'][0]

            return urls, distances

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image search failed: {str(e)}")

    def search_text(self, query_text: str, top_k: int = 5):
        """
        Search for relevant images given a text query with improved error handling.
        """
        try:
            # Get text embedding
            query_embedding = self.get_embedding_from_api(
                {"query": query_text},
                "/embed_text"
            )

            # Perform vector search
            results = self.vector_db.query(
                query_embeddings=query_embedding.numpy().tolist(),
                n_results=top_k
            )

            # Extract URLs and distances
            urls = [result.get('url', 'Unknown') for result in results['metadatas'][0]]
            distances = results['distances'][0]

            return urls, distances

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Text search failed: {str(e)}")

    def _load_image_hashes_from_db(self):
        """
        Load image hashes from the database with error handling.
        """
        try:
            # Retrieve all metadata
            all_metadata = self.vector_db.get(include=["metadatas"])["metadatas"]

            # Extract image hashes
            image_hashes = {
                metadata["image_hash"]
                for metadata in all_metadata
                if "image_hash" in metadata
            }

            return image_hashes

        except Exception as e:
            # If loading fails, start with an empty set
            print(f"Warning: Failed to load image hashes: {e}")
            return set()
