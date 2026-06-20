import os
import logging

# Configure logger
logger = logging.getLogger("traffic_db")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dotenv_path = os.path.join(project_root, ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
    else:
        load_dotenv()
except ImportError:
    pass

try:
    from pymongo import MongoClient
    _pymongo_available = True
except ImportError:
    _pymongo_available = False
    logger.warning("pymongo is not installed. MongoDB operations will be skipped.")

_mongo_client = None
DB_NAME = "traffic_support_db"
COLLECTION_NAME = "prediction_logs"

def get_mongo_client():
    """
    Returns a connected MongoClient instance, or None if MongoDB is not configured,
    not installed, or cannot be reached.
    """
    global _mongo_client
    
    if not _pymongo_available:
        return None
        
    # If already initialized, check if still connected
    if _mongo_client is not None:
        try:
            _mongo_client.admin.command('ping')
            return _mongo_client
        except Exception:
            logger.warning("Existing MongoDB connection lost. Attempting to reconnect...")
            try:
                _mongo_client.close()
            except Exception:
                pass
            _mongo_client = None

    uri = os.environ.get("MONGODB_URI")
    if not uri:
        logger.info("MONGODB_URI is not set in environment variables. Using file-based logs.")
        return None

    try:
        # Set a short server selection timeout (e.g. 3 seconds) so we don't block the API long if offline
        client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        # Test connection
        client.admin.command('ping')
        _mongo_client = client
        logger.info("Successfully connected to MongoDB Atlas!")
        return _mongo_client
    except Exception as e:
        logger.warning(f"Failed to connect to MongoDB: {e}. Falling back to file-based logs.")
        return None

def save_prediction_log(record: dict) -> bool:
    """
    Inserts a single prediction record into MongoDB.
    Returns True if successfully saved, False otherwise.
    """
    client = get_mongo_client()
    if client is None:
        return False
        
    try:
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        # Insert a copy to avoid modifying the original dictionary (e.g., adding _id in place)
        collection.insert_one(record.copy())
        logger.info("Successfully saved prediction log to MongoDB Atlas.")
        return True
    except Exception as e:
        logger.error(f"Error saving prediction log to MongoDB: {e}")
        return False

def fetch_prediction_history(limit: int = 200) -> list:
    """
    Fetches the latest prediction records from MongoDB, sorted descending by timestamp.
    Returns a list of dict records (excluding MongoDB _id), or None if database is unavailable.
    """
    client = get_mongo_client()
    if client is None:
        return None
        
    try:
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        # Fetch, sort by timestamp descending, limit, and remove '_id'
        cursor = collection.find({}, {'_id': 0}).sort('timestamp', -1).limit(limit)
        return list(cursor)
    except Exception as e:
        logger.error(f"Error fetching prediction logs from MongoDB: {e}")
        return None
