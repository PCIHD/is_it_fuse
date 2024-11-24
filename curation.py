import glob
import os
from multiprocessing import Pool, cpu_count
from pdfminer.high_level import extract_text, extract_pages
import fitz
import pdfplumber
import pdfminer
from pdfminer.layout import LTTextBox, LTTextLine
from tqdm import tqdm
import pandas as pd
from pypdf import PdfReader
import pdfplumber


def create_dataframe_record(filename: str):
    file_splits = filename.split("/")
    class_name = file_splits[4]
    dataset_type = file_splits[3]

    pages_text = ""
    try:
        for page_layout in extract_pages(filename):
            for element in page_layout:
                if isinstance(element, (LTTextBox, LTTextLine)):
                    pages_text = pages_text + "\n" + element.get_text()

    except Exception as e:
        print(f"Error extracting text: {e,filename}")
        return None
    return (pages_text, class_name, dataset_type, filename)


def process_files_with_multiprocessing(file_list):
    with Pool(processes=cpu_count()) as pool:  # Use all available CPUs
        # Use tqdm with the pool's `imap_unordered` for progress tracking
        results = list(
            tqdm(
                pool.imap(create_dataframe_record, file_list),
                total=len(file_list),
                desc="Processing PDFs",
            )
        )
    return results


if __name__ == "__main__":
    # Get the list of PDF files
    file_list = glob.glob("./data_foundation/raw/*/*/*")

    # Process files with multiprocessing
    results = process_files_with_multiprocessing(file_list)

    # Filter out None results (if any errors occurred)
    results = [res for res in results if res is not None]
    failures = ["_" for res in results if res is None]
    print(f"Failed at {len(failures)} PDFs")
    # Create DataFrame and save as Parquet
    df = pd.DataFrame(
        results, columns=["text", "class_name", "dataset_type", "filename"]
    )
    os.makedirs("./data_foundation/curated/", exist_ok=True)
    df.to_parquet("./data_foundation/curated/curated_dataset.parquet")
    df.to_excel("./data_foundation/curated/curated_dataset.xlsx")

    print("Data processing complete. Curated dataset saved.")
