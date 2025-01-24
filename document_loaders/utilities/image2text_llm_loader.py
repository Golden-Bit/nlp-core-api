import base64
import json
import os
from typing import List, Optional, Union, Iterator, Tuple
from PIL import Image
from langchain.document_loaders.base import BaseLoader
from langchain_core.documents import Document

from utilities.llm_as_function.image_description import ImageDescriptionFunction


class ImageDescriptionLoader(BaseLoader):
    """
    Loader that uses the ImageDescriptionFunction to analyze image files
    and generate descriptive documents. Supports both a directory or a specific image path.
    """

    def __init__(
        self,
        image_dir: Optional[str] = None,
        #image_path: Optional[str] = None,
        resize_to: Optional[Tuple[int, int]] = None,
        openai_api_key: str = "",
        postprocess: Optional[callable] = None,
        supported_formats: Optional[List[str]] = None
    ):
        """
        Initialize the ImageDescriptionLoader with the specified parameters.

        Args:
            image_dir: Optional; Path to the directory containing images.
            image_path: Optional; Path to a specific image file.
            resize_to: Optional; Tuple specifying (width, height) for resizing images.
            openai_api_key: API key for OpenAI.
            postprocess: Optional function to process the LLM output.
            supported_formats: List of supported image file extensions (e.g., [".jpg", ".png"]).
        """
        self.image_dir = image_dir
        #self.image_path = image_path
        self.resize_to = resize_to
        self.supported_formats = supported_formats or [".jpg", ".png", ".jpeg"]
        self.image_description_function = ImageDescriptionFunction(
            openai_api_key=openai_api_key,
            postprocess=postprocess
        )

        #if not image_dir and not image_path:
        #    raise ValueError("You must provide either 'image_dir' or 'image_path'.")

    def _get_image_files(self) -> List[str]:
        """
        Retrieves a list of supported image files from the directory or single file.
        """
        if self.image_dir:
            if os.path.splitext(self.image_dir)[1].lower() in self.supported_formats:
                return [self.image_dir]
            else:
                raise ValueError(f"Unsupported image format: {self.image_dir}")

        #if self.image_path:
        #    return [
        #        os.path.join(self.image_path, f)
        #        for f in os.listdir(self.image_path)
        #        if os.path.splitext(f)[1].lower() in self.supported_formats
        #    ]

        return []

    def _resize_image(self, image_path: str) -> str:
        """
        Resize an image to the specified dimensions and save it temporarily.

        Args:
            image_path: Path to the original image.

        Returns:
            Path to the resized image.
        """
        if not self.resize_to:
            return image_path

        resized_image_path = os.path.join(
            os.path.dirname(image_path), f"resized_{os.path.basename(image_path)}"
        )

        with Image.open(image_path) as img:
            img = img.resize(self.resize_to, Image.Resampling.LANCZOS)  # Use LANCZOS instead of ANTIALIAS
            img.save(resized_image_path)

        return resized_image_path

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazily load images and generate descriptive Documents.

        Yields:
            Documents with descriptive content for each image.
        """
        image_files = self._get_image_files()

        for image_path in image_files:
            try:
                # Resize image if needed
                processed_image_path = self._resize_image(image_path)

                # Encode image to base64
                image_base64 = self.image_description_function.encode_image_to_base64(processed_image_path)

                # Generate the description
                human_content = [
                    {"type": "text", "text": "Descrivi l'immagine fornita. Non generare analisi mediche."},
                    {"type": "image_url", "image_url": {"url": image_base64, "detail": "auto"}}
                ]

                description = self.image_description_function.execute(human_content)

                # Create a Document
                metadata = {
                    "source": image_path,
                    "filename": os.path.basename(image_path)
                }
                document = Document(page_content=description, metadata=metadata)

                yield document

            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    def load(self) -> List[Document]:
        """
        Eagerly load all image descriptions as Documents.

        Returns:
            List of Documents with descriptive content.
        """
        return list(self.lazy_load())


if __name__ == "__main__":

    def custom_postprocess(output: str) -> str:
        """
        Postprocessing function to extract the content of "descrizione_frame" from the output.
        """
        start_tag = "<attribute=frame_description|"
        end_tag = "| attribute=frame_description>"

        start_idx = output.find(start_tag)
        end_idx = output.find(end_tag)

        if start_idx == -1 or end_idx == -1:
            raise ValueError("Invalid format: Missing frame description markers.")

        json_content = output[start_idx + len(start_tag):end_idx].strip()
        try:
            parsed_content = json.loads(json_content)
            return parsed_content.get("descrizione_frame", "")
        except json.JSONDecodeError:
            raise ValueError("Error parsing JSON content.")

    # Initialize the loader with your image directory or single image path
    loader = ImageDescriptionLoader(
        image_dir="C:\\Users\\Golden Bit\\Downloads\\img.jpg",  # Set to None to use a single image
        #image_path="C:\\Users\\Golden Bit\\Downloads\\img.jpg",  # Replace with an actual image path
        resize_to=(256, 256),  # Resize images to 256x256
        openai_api_key="...",
        postprocess=custom_postprocess,  # Optional postprocessing function
    )

    # Load the documents
    documents = loader.load()

    print("Generated Descriptions:")
    for doc in documents:
        print(f"Filename: {doc.metadata.get('filename')}")
        print(f"Description: {doc.page_content}\n")
        print(f"Metadata: {doc.metadata}\n")
        print("---")
