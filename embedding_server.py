import io
import threading
from typing import List

import nest_asyncio
import requests
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor

# Initialize FastAPI
app = FastAPI(title="Embedding Server")

# Initialize the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Function to compute embeddings from text
def get_text_embedding(text: str) -> torch.Tensor:
    """
    Compute the embedding for a given text using the CLIP model.

    Args:
        text (str): The input text to compute the embedding for.

    Returns:
        torch.Tensor: The computed text embedding as a PyTorch tensor.
    """
    inputs = processor(text=text, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)
    return text_embedding


# Function to compute embeddings from image
def get_image_embedding(image_url: str) -> torch.Tensor:
    """
    Compute the embedding for an image given its URL using the CLIP model.

    Args:
        image_url (str): The URL of the image to compute the embedding for.

    Returns:
        torch.Tensor: The computed image embedding as a PyTorch tensor.
        str: An error message if the image cannot be processed.
    """
    try:
        # Fetch the image from the URL
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))

        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            image_embedding = model.get_image_features(**inputs)
        return image_embedding
    except Exception as e:
        return str(e)


# Define request and response models
class TextRequest(BaseModel):
    """
    Request model for text embedding.

    Attributes:
        query (str): The input text query to compute the embedding for.
    """
    query: str


class ImageRequest(BaseModel):
    """
    Request model for image embedding.

    Attributes:
        image_url (str): The URL of the image to compute the embedding for.
    """
    image_url: str


class EmbeddingResponse(BaseModel):
    """
    Response model for embedding results.

    Attributes:
        embedding (List[float]): The computed embedding as a list of floats.
    """
    embedding: List[float]


# Endpoint for text embedding
@app.post("/embed_text", response_model=EmbeddingResponse)
async def embed_text(request: TextRequest):
    """
    Endpoint to compute the embedding for a given text query.

    Args:
        request (TextRequest): The request containing the text query.

    Returns:
        EmbeddingResponse: The computed text embedding as a list of floats.
    """
    embedding = get_text_embedding(request.query)
    return EmbeddingResponse(embedding=embedding.numpy().tolist()[0])  # 1 x 512 -> 512


# Endpoint for image embedding
@app.post("/embed_image", response_model=EmbeddingResponse)
async def embed_image(request: ImageRequest):
    """
    Endpoint to compute the embedding for an image given its URL.

    Args:
        request (ImageRequest): The request containing the image URL.

    Returns:
        EmbeddingResponse: The computed image embedding as a list of floats.
        JSONResponse: An error response if the image cannot be processed.
    """
    embedding = get_image_embedding(request.image_url)
    if isinstance(embedding, str):  # Check if there was an error
        return JSONResponse(status_code=400, content={"message": embedding})

    return EmbeddingResponse(embedding=embedding.numpy().tolist()[0])  # 1 x 512 -> 512


# Run the server asynchronously within the notebook
def run():
    """
    Run the FastAPI server asynchronously.

    This function applies `nest_asyncio` to allow the server to run in a Jupyter notebook
    and starts the server using `uvicorn`.

    Returns:
        None
    """
    nest_asyncio.apply()  # This is necessary to allow uvicorn to run in a Jupyter notebook
    uvicorn.run(app, host="0.0.0.0", port=8000)
