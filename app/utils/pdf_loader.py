from pathlib import Path
import fitz
import logging

class MedicalPDFLoader:
    def __init__(self, file_path: str):

        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(f"Error: {file_path} doesn't exist")
        
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Error: {file_path} isn't a PDF file")

        self.file_path = file_path  


    def _read_pdf(self):

        try:
            with fitz.open(self.file_path) as doc:
                
                logging.info(f"Reading {self.file_path}")

                logging.info("Extracting paragraphs from document")
                all_blocks = (
                    block for page in doc
                    for block in page.get_text("blocks") # read paragraphs instead of lines
                )
                
                logging.info("Extracting text from paragraphs")
                paragraphs = [
                    block[4].strip()
                    for block in all_blocks
                    if block[6] == 0
                ]

                logging.info(f"Finish reading {self.file_path}. Extracted {len(paragraphs)} paragraphs")

                return paragraphs

        except Exception as e:
            logging.error(f"Error processing {self.file_path}: {e}")
