import json
from typing import Optional, Callable

from utilities.llm_as_function.base import LLMFunctionBase
import base64

# ---------------------------------------------------------------------------
# System prompt esteso per ImageDescriptionFunction
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT_IMAGE = """
Sei un assistente visivo professionale.  
Obiettivo: produrre **descrizioni straordinariamente dettagliate** di ogni immagine.

ISTRUZIONI:
1. Elenca **tutti** gli elementi visibili: oggetti, persone, animali, sfondi, texture, materiali.
2. Trascrivi **integralmente** qualunque testo, numero, logo, simbolo, unità di misura o codice.
3. Riporta date, valute e percentuali **esattamente** come appaiono.
4. Se un carattere non è leggibile, usa “[illeggibile]”.
5. Mantieni un tono tecnico e neutro, **senza diagnosi** né giudizi soggettivi.
"""
#"""
#6. Incapsula la risposta in:
#   <attribute=frame_description| {"descrizione_frame": "<TESTO>"} | attribute=frame_description>
#"""



# Extend the base class to create a specific use case
class ImageDescriptionFunction(LLMFunctionBase):
    """
    A specialized class for describing images with the LLM.
    """
    def __init__(self, openai_api_key: str, postprocess: Optional[Callable[[str], str]] = None):
        super().__init__(
            name="Image Description",
            description="Generates qualitative and aesthetic descriptions of images, maintaining consistency across frames.",
            system_message=DEFAULT_SYSTEM_PROMPT_IMAGE,
            openai_api_key=openai_api_key,
            postprocess=postprocess
        )

    @staticmethod
    def encode_image_to_base64(image_path: str) -> str:
        """
        Encodes an image file to a base64 string.
        """
        with open(image_path, "rb") as image_file:
            return "data:image/jpeg;base64," + base64.b64encode(image_file.read()).decode("utf-8")


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

    api_key = "...."

    image_description_function = ImageDescriptionFunction(
        openai_api_key=api_key,
        postprocess=custom_postprocess
    )

    # Path to the image file
    image_path = "C:\\Users\\Golden Bit\\Downloads\\img.jpg"  # Replace with an actual image path
    image_base64 = image_description_function.encode_image_to_base64(image_path)

    previous_descriptions = ["Un paesaggio montano con alberi innevati.", "Un fiume che scorre tra le colline."]

    human_content = [
        {"type": "text",
         "text": "Analizza il frame seguente. Tieni conto delle descrizioni dei frame precedenti fornite. Non generare analisi mediche. Cerca di mantenere coerenza con le descrizioni precedenti."},
    ]

    for idx, desc in enumerate(previous_descriptions):
        human_content.append({"type": "text", "text": f"Descrizione frame precedente {idx + 1}: {desc}"})

    human_content.append({"type": "image_url", "image_url": {"url": image_base64, "detail": "auto"}})

    result = image_description_function.execute(human_content)
    print("Output:", result)
