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

        paragraphs = []

        try:
            with fitz.open(self.file_path) as doc:
                
                logging.info(f"Reading {self.file_path}")

                for num_page in range(len(doc)):
                    logging.info(f"--- Start reading page {num_page + 1} ---")

                    page = doc.load_page(num_page)

                    logging.info("Extracting paragraphs")
                    blocks = page.get_text("blocks") # read paragraphs instead of lines

                    for block in blocks:
                        if block[6] == 0: # if the block is a text
                            logging.info(f"Extracting text from block {block[5]}")
                            text = block[4].strip() # Remove initial and final spaces
                            paragraphs.append(text)

                    logging.info(f"--- End reading page {num_page + 1} ---")

                logging.info(f"Finish reading {self.file_path}")

        except Exception as e:
            logging.error(f"Error processing {self.file_path}: {e}")

        return paragraphs
