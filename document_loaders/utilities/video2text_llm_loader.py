import base64
import json
import os
import uuid
import tempfile
from typing import List, Optional, Iterator, Tuple

import cv2
from PIL import Image

from langchain.document_loaders.base import BaseLoader
from langchain_core.documents import Document

from langchain_openai import ChatOpenAI
from langchain.schema.messages import SystemMessage, HumanMessage
from langchain_core.messages import AIMessage

# Definizione del prompt di sistema per l'analisi video
SYSTEM_PROMPT = """
Sei un assistente virtuale specializzato nell'analisi visiva di frame estratti da un video. Ti verranno forniti dei frame sotto forma di immagini (in base64). 
Ad ogni chiamata ti verrà passato un nuovo frame insieme alla storia dei frame precedenti già descritti. Il tuo compito è:
1. Fornire una descrizione qualitativa del frame corrente, coerente con le immagini fornite e che fornisca dettagli rilevanti dal punto di vista visivo, come se fosse un'analisi professionale dell'immagine, ma non medica e senza alcuna pretesa diagnostica.
2. La descrizione deve essere racchiusa tra i seguenti marker speciali, per consentirne il parsing: 
<attribute=frame_description| {"descrizione_frame": "..."} | attribute=frame_description>

Dopo aver analizzato tutti i frame, dovrai fornire una descrizione finale del video. La descrizione finale sarà prodotta in un'ultima chiamata senza nuovi frame, basandoti sulle descrizioni dei singoli frame. Anche questa descrizione finale deve essere fornita racchiusa tra speciali marker:
<attribute=final_description| {"descrizione_finale": "..."} | attribute=final_description>

Ricorda: Ogni volta che analizzi un nuovo frame devi considerare la descrizione dei frame precedenti che ti verrà fornita come chat history. Non devi ripetere le descrizioni precedenti, ma devi tenerle in considerazione per mantenere coerenza nel tempo.
"""

class VideoDescriptionLoader(BaseLoader):
    """
    Loader that uses a video analysis procedure to process video files,
    extract frames, and generate descriptive Documents. The loader supports either a single video file
    or a directory of video files, extracting a subset of frames (with optional resizing) and generating:
      - A description for each frame (with context dei frame precedenti)
      - A final description of the video aggregando tutte le analisi
    """
    def __init__(
        self,
        file_path: Optional[str] = None,
        model_name: str = "gpt-4o",
        video_dir: Optional[str] = None,
        resize_to: Optional[Tuple[int, int]] = None,
        num_frames: Optional[int] = None,
        frame_rate: Optional[int] = None,
        openai_api_key: str = "",
        #postprocess: Optional[callable] = None,
        supported_formats: Optional[List[str]] = None
    ):
        """
        Initialize the VideoDescriptionLoader with the specified parameters.

        Args:
            file_path: Optional; Path to a single video file.
            video_dir: Optional; Path to a directory containing video files.
            resize_to: Optional; Tuple specifying (width, height) for resizing extracted frames.
            num_frames: Optional; If provided, extract this number of frames uniformly distributed.
            frame_rate: Optional; If provided, extract frames at the specified frame rate.
            openai_api_key: API key for OpenAI.
            postprocess: Optional function to process the LLM output.
            supported_formats: List of supported video file extensions (e.g., [".mp4", ".avi", ".mov"]).
        """
        self.file_path = file_path
        self.video_dir = video_dir
        self.resize_to = resize_to
        self.num_frames = num_frames
        self.frame_rate = frame_rate
        #self.postprocess = postprocess
        self.supported_formats = supported_formats or [".mp4", ".avi", ".mov"]

        # Initialize the Chat model with the given API key and parameters.
        self.chat = ChatOpenAI(
            model_name=model_name,
            temperature=0.25,
            max_tokens=2048,
            openai_api_key=openai_api_key
        )

    def _get_video_files(self) -> List[str]:
        """
        Retrieves a list of supported video files from the provided video_path or video_dir.
        """
        files = []
        if self.file_path:
            ext = os.path.splitext(self.file_path)[1].lower()
            if ext in self.supported_formats:
                files.append(self.file_path)
            else:
                raise ValueError(f"Unsupported video format: {self.file_path}")
        elif self.video_dir:
            for f in os.listdir(self.video_dir):
                if os.path.splitext(f)[1].lower() in self.supported_formats:
                    files.append(os.path.join(self.video_dir, f))
        return files

    def _extract_frames(self, video_file: str) -> List[str]:
        """
        Extract frames from the video file.
        If num_frames is provided, extract that many frames uniformly distributed.
        If frame_rate is provided, extract frames at that rate.
        Additionally, resize each frame to the specified dimensions if resize_to is set.
        Returns a list of paths to the extracted (and possibly resized) frame images.
        """
        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Decide which frames to extract
        if self.num_frames is not None:
            frame_indices = [int(i * total_frames / self.num_frames) for i in range(self.num_frames)]
        elif self.frame_rate is not None:
            frame_step = int(fps / self.frame_rate) if self.frame_rate <= fps else 1
            frame_indices = list(range(0, total_frames, frame_step))
        else:
            # Default: extract 5 frames uniformly
            default_num = 5
            frame_indices = [int(i * total_frames / default_num) for i in range(default_num)]

        frame_paths = []
        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Unable to read frame at index {idx} in video {video_file}.")
                break

            # Resize frame if required
            if self.resize_to:
                width, height = self.resize_to
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            # Save the frame to a temporary file
            tmp_dir = tempfile.mkdtemp()
            frame_path = os.path.join(tmp_dir, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        cap.release()
        return frame_paths

    def _frame_to_base64(self, frame_path: str) -> str:
        """
        Encode the frame image at frame_path to a base64 string.
        """
        with open(frame_path, "rb") as f:
            return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazily load video files and generate descriptive Documents.
        For each video, it:
          - Extracts frames.
          - Processes each frame sequentially with il contesto delle analisi precedenti.
          - Richiede infine una descrizione finale del video.
        Yields:
            Documents with descriptive content for each video.
        """
        video_files = self._get_video_files()
        MAX_PREVIOUS_DESCRIPTIONS = 100  # Limite per la storia delle descrizioni precedenti

        for video_file in video_files:
            # Estrai i frame dal video
            frame_paths = self._extract_frames(video_file)

            # Inizializza la conversazione con il prompt di sistema
            system_message = SystemMessage(content=SYSTEM_PROMPT)
            messages = [system_message]
            frame_descriptions = []

            # Analizza ogni frame
            for i, frame_path in enumerate(frame_paths):
                frame_b64 = self._frame_to_base64(frame_path)

                # Prepara il contenuto del messaggio umano con le descrizioni precedenti
                human_content = [
                    {"type": "text", "text": "Analizza il frame seguente. Tieni conto delle descrizioni dei frame precedenti fornite. Non generare analisi mediche. Cerca di mantenere coerenza con le descrizioni precedenti."}
                ]
                for idx, desc in enumerate(frame_descriptions[-MAX_PREVIOUS_DESCRIPTIONS:]):
                    human_content.append({"type": "text", "text": f"Descrizione frame precedente {idx+1}: {desc}"})
                human_content.append({"type": "image_url", "image_url": {"url": frame_b64, "detail": "auto"}})

                human_message = HumanMessage(content=human_content)
                # Invia la richiesta al modello
                response = self.chat(messages + [human_message])
                ai_response = response.content

                # Parsing della risposta per estrarre la descrizione del frame
                start_tag = "<attribute=frame_description|"
                end_tag = "| attribute=frame_description>"
                start_idx = ai_response.find(start_tag)
                end_idx = ai_response.find(end_tag)
                if start_idx == -1 or end_idx == -1:
                    raise ValueError("La risposta del modello non contiene la descrizione formattata correttamente per il frame.")

                json_str = ai_response[start_idx + len(start_tag):end_idx].strip()
                try:
                    desc_dict = json.loads(json_str)
                    desc_frame = desc_dict.get("descrizione_frame", "")
                except json.JSONDecodeError:
                    raise ValueError("Errore nel parsing del JSON per la descrizione del frame.")

                # Aggiorna la storia della conversazione e le descrizioni dei frame
                messages.append(human_message)
                messages.append(AIMessage(content=ai_response))
                frame_descriptions.append(desc_frame)

            # Genera la descrizione finale del video
            final_human_content = [
                {"type": "text", "text": "Genera la descrizione finale del video basandoti sulle descrizioni dei frame precedenti. Non analisi mediche, ma solo qualitative ed estetiche. Fornisci la descrizione finale racchiusa nei tag richiesti."}
            ]
            for idx, d in enumerate(frame_descriptions):
                final_human_content.append({"type": "text", "text": f"Descrizione frame {idx+1}: {d}"})
            final_human_message = HumanMessage(content=final_human_content)
            final_response = self.chat(messages + [final_human_message])
            final_text = final_response.content

            final_start_tag = "<attribute=final_description|"
            final_end_tag = "| attribute=final_description>"
            fs_idx = final_text.find(final_start_tag)
            fe_idx = final_text.find(final_end_tag)
            if fs_idx == -1 or fe_idx == -1:
                raise ValueError("La risposta del modello non contiene la descrizione finale formattata correttamente.")
            final_json_str = final_text[fs_idx + len(final_start_tag):fe_idx].strip()
            try:
                final_desc_dict = json.loads(final_json_str)
                final_description = final_desc_dict.get("descrizione_finale", "")
            except json.JSONDecodeError:
                raise ValueError("Errore nel parsing del JSON per la descrizione finale.")

            # Prepara i metadati e crea il Document
            metadata = {
                "source": video_file,
                "frame_descriptions": frame_descriptions
            }
            document = Document(page_content=final_description, metadata=metadata)
            yield document

    def load(self) -> List[Document]:
        """
        Eagerly load all video descriptions as Documents.
        Returns:
            List of Documents with descriptive content.
        """
        return list(self.lazy_load())

    def custom_postprocess(self, output: str) -> str:
        start_tag = "<attribute=frame_description|"
        end_tag = "| attribute=frame_description>"
        start_idx = output.find(start_tag)
        end_idx = output.find(end_tag)
        if start_idx == -1 or end_idx == -1:
            raise ValueError("Formato non corretto: marker mancanti.")
        json_content = output[start_idx + len(start_tag):end_idx].strip()
        try:
            parsed_content = json.loads(json_content)
            return parsed_content.get("descrizione_frame", "")
        except json.JSONDecodeError:
            raise ValueError("Errore nel parsing del contenuto JSON.")

if __name__ == "__main__":
    # Esempio di utilizzo:
    # Definisci, se necessario, una funzione di postprocess (qui non obbligatoria, poiché il parsing è già gestito)


    # Inizializza il loader con il percorso del video (oppure specifica video_dir) e altri parametri desiderati
    loader = VideoDescriptionLoader(
        file_path="C:\\Users\\info\\OneDrive\\Desktop\\work_space\\repositories\\nlp-core-api\\document_loaders\\utilities\\test_video.mp4",  # Sostituire con il percorso del video
        resize_to=(256, 256),      # Dimensione per il resize dei frame
        num_frames=5,              # Numero di frame da estrarre (opzionale)
        # frame_rate=2,           # In alternativa, specificare un frame_rate
        openai_api_key="...",      # Inserisci la tua API key OpenAI
        #postprocess=custom_postprocess  # Funzione di postprocess opzionale
    )

    # Carica i Document generati
    documents = loader.load()

    print("Descrizioni Generate per il Video:")
    for doc in documents:
        print(f"Video Source: {doc.metadata.get('source')}")
        print("Frame Descriptions:")
        for idx, desc in enumerate(doc.metadata.get("frame_descriptions", [])):
            print(f"  Frame {idx+1}: {desc}")
        print(f"Final Video Description: {doc.page_content}\n")
        print(f"Metadata: {doc.metadata}\n")
        print("---")
