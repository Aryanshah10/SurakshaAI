import os

from dotenv import load_dotenv

from officers import OFFICERS

load_dotenv()


def authenticate_user(username: str, password: str):

    officer = OFFICERS.get(username)

    if officer is None:
        return None

    actual_password = os.getenv(officer["password_key"])

    if actual_password != password:
        return None

    return officer