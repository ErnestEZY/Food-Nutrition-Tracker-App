import streamlit as st
import pymongo
from pymongo.errors import PyMongoError, NetworkTimeout, ServerSelectionTimeoutError
import time

# MongoDB Connection with Retry Mechanism
@st.cache_resource
def init_connection():
    max_retries = 3
    try:
        mongodb_uri = st.secrets["MONGODB_URI"]
    except KeyError:
        st.error("MONGODB_URI not found in secrets. Please check your .streamlit/secrets.toml or Streamlit Cloud Secrets.")
        return None
    
    if not mongodb_uri:
        st.error("MONGODB_URI is empty. Please provide a valid MongoDB connection string.")
        return None
    
    for attempt in range(max_retries):
        try:
            client = pymongo.MongoClient(
                mongodb_uri,
                serverSelectionTimeoutMS=5000
            )
            client.admin.command('ismaster')
            db = client["FoodNutri"]
            return db
        except (PyMongoError, NetworkTimeout, ServerSelectionTimeoutError) as e:
            st.error(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                st.error("Failed to connect to MongoDB. Please check your connection.")
                return None
            time.sleep(2)

# Initialize database collections
db = init_connection()
food_collection = db["food_data"] if db is not None else None
daily_log_collection = db["daily_logs"] if db is not None else None
users_collection = db["users"] if db is not None else None

def safe_mongodb_operation(operation, error_message="Operation failed"):
    try:
        return operation()
    except (PyMongoError, NetworkTimeout, ServerSelectionTimeoutError) as e:
        st.error(f"{error_message}: {e}")
        if st.button("Retry Operation"):
            st.experimental_rerun()
        return None