import os
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path

import PyPDF2
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3 import Retry
import urllib.request

data_source = pd.ExcelFile("./meta_data/DataSet.xlsx")


# few samples will be lost due to connection issues, can be resolved using http adapter with a retry mechanism
def fetch_and_store_file(file_name: dict) -> bool:
    """Fetches one pdf from the provided dict record

    Parameters
    ----------
    file_name: dict
        A dictionary containing the following keys:
            datasheet_link
            dataset_section
            target_col

    Returns
    -------
    bool
        Indicating success or failure of fetch operation
    """
    file_name_url = file_name.get("datasheet_link")
    if not (file_name_url.startswith("https") or file_name_url.startswith("http")):
        file_name_url = "http:" + file_name_url
    file_write_path_root = "./data_foundation/raw"
    file_write_path = Path(
        file_write_path_root,
        file_name.get("dataset_section"),
        file_name.get("target_col"),
    )
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0 Safari/537.36"
        }
        response = requests.get(
            file_name_url,
            stream=True,
            headers=headers,
            allow_redirects=True,
            timeout=500,
        )
        response.raise_for_status()
        file_name_url_reduced = file_name_url[-30:]
        file_name_url_reduced = file_name_url_reduced.replace("/", "-")
        if not file_name_url_reduced.endswith(".pdf"):
            file_name_url_reduced += ".pdf"
        with open(
            os.path.join(Path(file_write_path, Path(file_name_url_reduced))), "wb"
        ) as pdf_file:
            for chunk in response.iter_content(chunk_size=1024):
                pdf_file.write(chunk)
        return True
    except Exception as e:
        print(e)
        print(os.path.join(Path(file_write_path, Path(file_name_url.split("/")[-1]))))
        print(file_name_url)
        return False


for sheet in data_source.sheet_names:
    sheet_data = data_source.parse(sheet)
    sheet_data = sheet_data[sheet_data["datasheet_link"].str.len() > 1]
    sheet_data["dataset_section"] = sheet
    file_write_path_root = "./data_foundation/raw"
    for classes in tqdm(sheet_data["target_col"].unique()):
        file_write_path = Path(file_write_path_root, sheet, classes)
        if not file_write_path.exists():
            os.makedirs(file_write_path)
    sheet_records = sheet_data.to_dict(orient="records")
    # for record in tqdm(sheet_records):
    #     fetch_and_store_file(record)

    results = []
    with ThreadPoolExecutor(max_workers=200) as executor:
        # results = list(executor.map(fetch_and_store_file, sheet_records))
        futures = [
            executor.submit(fetch_and_store_file, record) for record in sheet_records
        ]

        # Use tqdm to show progress bar
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Downloading"
        ):
            results.append(future.result())
