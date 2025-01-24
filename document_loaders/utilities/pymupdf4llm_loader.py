from typing import List, Optional, Union
from langchain.document_loaders.base import BaseLoader
from langchain.docstore.document import Document
import pymupdf4llm


class PyMuPDFLoader(BaseLoader):
    """Loader that uses pymupdf4llm to load PDF documents."""

    def __init__(
            self,
            file_path: str,
            pages: Optional[List[int]] = None,
            hdr_info: Optional[str] = None,
            extract_images: Optional[str] = True,
            write_images: bool = False,
            embed_images: bool = False,
            image_path: Optional[str] = None,
            image_format: str = "png",
            image_size_limit: float = 0.00,
            force_text: bool = True,
            page_chunks: bool = True,
            margins: Optional[Union[List[float], float]] = [0,0,0,0],
            dpi: int = 300,
            page_width: int = 1224,
            page_height: Optional[int] = None,
            table_strategy: str = "lines_strict",
            graphics_limit: Optional[int] = None,
            fontsize_limit: int = 3,
            ignore_code: bool = False,
            extract_words: bool = False,
            show_progress: bool = False,
    ):
        """
        Initialize the PymuPDFLoader with the specified parameters.
        """
        self.file_path = file_path
        self.pages = pages
        self.hdr_info = hdr_info
        self.extract_images = extract_images
        self.write_images = write_images
        self.embed_images = embed_images
        self.image_path = image_path
        self.image_format = image_format
        self.image_size_limit = image_size_limit
        self.force_text = force_text
        self.page_chunks = page_chunks
        self.margins = margins or [0, 0, 0, 0]
        self.dpi = dpi
        self.page_width = page_width
        self.page_height = page_height
        self.table_strategy = table_strategy
        self.graphics_limit = graphics_limit
        self.fontsize_limit = fontsize_limit
        self.ignore_code = ignore_code
        self.extract_words = extract_words
        self.show_progress = show_progress

    def load(self) -> List[Document]:
        """
        Load the PDF file and return a list of Documents.
        """
        # Use pymupdf4llm to extract content
        try:
            md_output = pymupdf4llm.to_markdown(
                self.file_path,
                pages=self.pages,
                hdr_info=self.hdr_info,
                #extract_images=self.extract_images,
                write_images=self.write_images,
                embed_images=self.embed_images,
                image_path=self.image_path,
                image_format=self.image_format,
                image_size_limit=self.image_size_limit,
                force_text=self.force_text,
                page_chunks=self.page_chunks,
                margins=self.margins,
                dpi=self.dpi,
                page_width=self.page_width,
                page_height=self.page_height,
                table_strategy=self.table_strategy,
                graphics_limit=self.graphics_limit,
                fontsize_limit=self.fontsize_limit,
                ignore_code=self.ignore_code,
                extract_words=self.extract_words,
                show_progress=self.show_progress,
            )

            # Process md_output to create Documents
            documents = []

            if self.page_chunks:
                # md_output is a list of page content
                for page_content in md_output:
                    #print("#" * 120)
                    #del page_content["text"]
                    #print(page_content)
                    #print("#" * 120)
                    page_text = page_content.get('text', '')
                    page_number = page_content.get('metadata', {}).get('page', None)
                    metadata = {'source': self.file_path}
                    if page_number is not None:
                        metadata['page'] = page_number
                    doc = Document(page_content=page_text, metadata=metadata)
                    documents.append(doc)
            else:
                # md_output is a string
                metadata = {'source': self.file_path}
                doc = Document(page_content=md_output, metadata=metadata)
                documents.append(doc)

            return documents

        except Exception as e:
            raise Exception(f"Error loading file {self.file_path}: {e}")


if __name__ == "__main__":
    # Initialize the loader with your PDF file and desired parameters
    loader = PyMuPDFLoader(
        file_path="source_data_dir/subdirectory/Giulio-36916281-02-10-2024.pdf",
        pages=None,  # Process all pages
        page_chunks=True,  # Split content by pages
        show_progress=True,  # Show progress during processing
        embed_images=False
    )

    # Load the documents
    documents = loader.load()

    print(documents)

    # Process the documents as needed
    for doc in documents:
        print(f"Page Number: {doc.metadata.get('page')}")
        print(f"Content:\n{doc.page_content}\n")
        print(f"Metadata: {doc.metadata}\n")
        print("---\n")