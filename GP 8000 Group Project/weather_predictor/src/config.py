import os

# --- Dataset Configuration ---
# Default dataset ID for weather data (e.g., Daily Weather Observations)
# Replace with the actual ID you intend to use primarily.
DEFAULT_DATASET_ID = "d_03bb2eb67ad645d0188342fa74ad7066"

# --- API Configuration ---
# Base URL for open data API (used for dataset download polling)
API_BASE_URL_OPEN = "https://api-open.data.gov.sg"
# Base URL for production API (used for dataset metadata - keep if needed)
API_BASE_URL_PRODUCTION = "https://api-production.data.gov.sg"
# API endpoint for the 24-hour weather forecast
API_24HR_FORECAST_URL = f"{API_BASE_URL_OPEN}/v2/real-time/api/twenty-four-hr-forecast?"
# Referer header often needed when interacting with data.gov.sg APIs programmatically
API_REFERER = 'https://colab.research.google.com' # Or a more generic/appropriate one if running locally

# --- Model & Training Configuration ---
# Default base directory for saving/loading models and stats
# Assumes it's one level up from the 'src' directory
DEFAULT_MODELS_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))

# Subdirectory names within the base directory
TRANSFORMER_MODEL_DIR_NAME = "transformer"
MLP_MODEL_DIR_NAME = "mlp"

# Default filenames
STATS_FILENAME = "stats.json"
TRANSFORMER_MODEL_FILENAME = "weather_transformer.pt"
MLP_MODEL_FILENAME = "mlp_model.pt"

# --- Derived Paths (Convenience) ---
def get_model_dir(base_dir, model_type_name):
    """Constructs the full path to a specific model's directory."""
    return os.path.join(base_dir, model_type_name)

def get_model_path(base_dir, model_type_name, model_filename):
    """Constructs the full path to a specific model file."""
    return os.path.join(base_dir, model_type_name, model_filename)

def get_stats_path(base_dir, model_type_name, stats_filename=STATS_FILENAME):
    """Constructs the full path to a model's statistics file."""
    return os.path.join(base_dir, model_type_name, stats_filename)
