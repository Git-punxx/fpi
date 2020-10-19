import sqlite3
from app_config import config_manager


headers = config_manager.categories.keys()


if __name__ == '__main__':
    print(headers)