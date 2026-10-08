import os
from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv()

# Access your config variables
REFERENCE_DIR = os.getenv("REFERENCE_DIR")
TARGET_DIR = os.getenv("TARGET_DIR")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_GPT_DEPLOYMENT = os.getenv("AZURE_GPT_DEPLOYMENT")
API_VERSION = os.getenv("API_VERSION")
GOLD_STANDARD_DIR = os.getenv("GOLD_STANDARD_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
WITHOUT_SEGMENTATION_OUTPUT_DIR = os.getenv("WITHOUT_SEGMENTATION_OUTPUT_DIR")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
YOLO_PREDICT_DIR = os.getenv("YOLO_PREDICT_DIR")
