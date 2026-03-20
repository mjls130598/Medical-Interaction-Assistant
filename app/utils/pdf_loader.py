from pathlib import Path
import re

class MedicalPDFLoader:
    def __init__(self, file_path: str):

        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(f"Error: {file_path} doesn't exist")
        
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Error: {file_path} isn't a PDF file")

        self.file_path = file_path      
