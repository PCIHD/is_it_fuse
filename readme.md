# is_it_fuse
- classification project to determine if the provide document is one of the classes or not

## How to setup
- create an environent using conda/ pip
- install requirements from the requirements.txt file
- start mlflow server
- For training run train.py
  - For training we must run ingestion.py and curation.py
  - Before running ingestion a excel file with urls and classes must be placed in a meta_data folder at project root.
  - after the files have been fetch and curated we are ready for training
  - Run train.py for training
- For inference run inference.py
  - Inference.py assumes there is
    - trained model exists in the models folder
    - a pdf file exists at a desired location
    - path to pdf file is provided as an argument to inference.py
  
## Folder structure

- data_foundation
  - raw
  - curated
- meta_data
- train.py
- inference.py
- ingestion.py
- curation.py



*formatted with black :)*
