import pymupdf
import pyttsx3
from pathlib import Path

speech_engine = pyttsx3.init()

def extract_selectable(pdf):
    '''
    Takes in a pdf file, then extracts all text content from the pdf file

    Args:
        pdf (string): the path to the pdf file
        
    Returns:
        extracted_text: the text extracted from the pdf file
    '''

    document_to_read = pymupdf.open(pdf)
    result = ""

    for page_no in range(document_to_read.page_count):
        curr_page = document_to_read.load_page(page_no)
        text_on_page = curr_page.get_text()
        result += text_on_page

    return result

def text_to_speech(text, path: str = "output.wav"):
    '''
    Takes in a string, then creates an audio file from the text

    Args:
        text (string): text to convert into an audio file
        path (string): path to save audio file
    
    Returns:
        output: absolute file path to audio file 
    '''
    speech_engine.save_to_file(text, path)
    speech_engine.runAndWait()

    return str(Path(path).resolve())
