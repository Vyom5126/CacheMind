"""
Settings read from the environment (or a .env file in the project root).
See .env.example for the full list.
"""

import os
from dotenv import load_dotenv

load_dotenv()

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "cold_storage"),
}
