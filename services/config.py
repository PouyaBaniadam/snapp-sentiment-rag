import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(PROJECT_ROOT, '..', 'data', 'train.csv')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'results.csv')

API_KEY = "AXN__oUlHMoSQLTzB35SJMzcYYhkQAJbjb_pnDLSx4gixeTXVQ8PJnezpGdcvPVfBjUdumeOVfxcdSf8Yc2vjw_FsFZ46IiIWpGmkecYd"
JWT_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzk1NTI4NDczLCJpYXQiOjE3NjM5OTI0NzMsImp0aSI6Ijk3ODQ1MTJlMzE1NzRmZjg4MzM3MDJlYzU2OTVhYjQwIiwidXNlcl9pZCI6IjEifQ.qnmfqIPrUigyV-C747GpjLhMyXigXYa5bos_8MDnT90"

BASE_URL = "http://127.0.0.1:8080"
GENERATE_ENDPOINT = f"{BASE_URL}/ai/v1/generate/text"
STATUS_CHECK_URL = f"{BASE_URL}/account/api_keys/"

MODEL_CONFIG = {
    "gpt-4o": {"char_limit": 4500, "priority": 1},
    "gpt-4o-mini": {"char_limit": 4000, "priority": 2},
    "gpt-4.1": {"char_limit": 5000, "priority": 3},
    "gpt-4.1-mini": {"char_limit": 4500, "priority": 4},
    "gpt-4.1-nano": {"char_limit": 4000, "priority": 5},
    "mistral-small-2503": {"char_limit": 3000, "priority": 6},
}

BATCH_POLL_INTERVAL_SECONDS = 5
MAX_RETRIES = 2
WAIT_SECONDS_IF_NO_MODEL = 60