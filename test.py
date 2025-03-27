import uuid
import pytest
import os
import shutil
import tempfile
from image_search_engine import ImageSearchEngine


# Define the base URL for the sample images
BASE_IMAGE_URL = "https://raw.githubusercontent.com/yavuzceliker/sample-images/main/images"


@pytest.fixture
def image_search_engine():
    """
    Fixture to create a temporary ImageSearchEngine instance with a unique collection for each test.
    """
    # Create a unique collection name
    collection_name = f"test_image_search_db_{uuid.uuid4().hex}"

    # Create the engine with this unique collection name
    temp_dir = './temp_db'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    engine = ImageSearchEngine(
        db_name=collection_name,
        api_url="http://localhost:8000",  # Adjust as needed
    )
    yield engine, collection_name
    del engine


def test_index_single_image(image_search_engine):
    """
    Test indexing a single image.
    """
    engine, _ = image_search_engine
    image_url = f"{BASE_IMAGE_URL}/image-1.jpg"
    engine.index_image(image_url)

    # Verify the image was indexed
    assert len(engine.image_hashes) == 1
    assert hash(image_url) in engine.image_hashes


def test_index_multiple_images(image_search_engine):
    """
    Test indexing multiple images.
    """
    engine, _ = image_search_engine
    for i in range(1, 16):
        image_url = f"{BASE_IMAGE_URL}/image-{i}.jpg"
        engine.index_image(image_url)

    # Verify 15 images were indexed
    assert len(engine.image_hashes) == 15


def test_index_duplicate_image(image_search_engine):
    """
    Test attempting to index the same image twice.
    """
    engine, _ = image_search_engine
    image_url = f"{BASE_IMAGE_URL}/image-1.jpg"
    engine.index_image(image_url)

    # Attempt to index the same image again should raise an HTTPException
    with pytest.raises(Exception) as excinfo:
        engine.index_image(image_url)

    assert "Image URL already indexed" in str(excinfo.value)


def test_retrieve_by_image(image_search_engine):
    """
    Test retrieving similar images by image URL.
    """
    engine, _ = image_search_engine
    # Index multiple images
    for i in range(1, 11):
        image_url = f"{BASE_IMAGE_URL}/image-{i}.jpg"
        engine.index_image(image_url)

    # Search for similar images to a previously indexed image
    search_url = f"{BASE_IMAGE_URL}/image-5.jpg"
    results, _ = engine.search_image(search_url, top_k=3)

    # Verify results
    assert len(results) > 0


def test_retrieve_by_text(image_search_engine):
    """
    Test retrieving images by text query.
    """
    engine, _ = image_search_engine
    # Index images
    images_data = [
        f"{BASE_IMAGE_URL}/image-1.jpg",
        f"{BASE_IMAGE_URL}/image-2.jpg", 
        f"{BASE_IMAGE_URL}/image-3.jpg"
    ]

    for url in images_data:
        engine.index_image(url)

    # Search for images with a text query
    results, _ = engine.search_text("sample text", top_k=2)

    # Verify results exist
    assert len(results) > 0


def test_save_and_load_db(image_search_engine):
    """
    Test saving and loading the database. Because it's persistent,
    we recreate the engine to simulate loading.
    """
    engine, collection_name = image_search_engine
    # Index some images
    for i in range(1, 6):
        image_url = f"{BASE_IMAGE_URL}/image-{i}.jpg"
        engine.index_image(image_url)

    old_hashes = engine.image_hashes.copy()
    del engine

    # Create a new instance and load the database
    new_engine = ImageSearchEngine(
        db_name=collection_name,
        api_url="http://localhost:8000",
    )

    # Ensure that 5 hashes from the new engine match the old ones
    assert set(old_hashes) == set(new_engine.image_hashes)


def test_delete_and_readd_image(image_search_engine):
    """
    Test deleting and re-adding an image (note: ChromaDB doesn't have a direct delete method, 
    so this test simulates the behavior).
    """
    engine, _ = image_search_engine
    image_url = f"{BASE_IMAGE_URL}/image-1.jpg"

    # Initial index
    engine.index_image(image_url)
    assert hash(image_url) in engine.image_hashes

    # Simulate deletion by removing from image_hashes
    engine.remove_image_from_index(image_url)
    assert hash(image_url) not in engine.image_hashes

    # Re-add the image
    engine.index_image(image_url)
    assert hash(image_url) in engine.image_hashes


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
    if os.path.exists('./temp_db'):
        shutil.rmtree('./temp_db')
