import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_env_vars():
    """
    Load environment variables from .env file.
    
    Returns:
        dict: Dictionary containing environment variables
    """
    # Check if .env file exists
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if not os.path.exists(env_path):
        logger.warning(f".env file not found at {env_path}. Using environment variables from system.")
    else:
        # Load environment variables from .env file
        load_dotenv(env_path)
        logger.info(f"Loaded environment variables from {env_path}")
    
    # Return dictionary with required environment variables
    env_vars = {
        "HF_TOKEN": os.getenv("HF_TOKEN"),
    }
    
    # Warn if HF_TOKEN is not set
    if not env_vars["HF_TOKEN"]:
        logger.warning("HF_TOKEN not set in .env file or environment variables")
        
    return env_vars