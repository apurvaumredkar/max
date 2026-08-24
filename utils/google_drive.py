"""
Standalone Google Drive helper. Not wired into the agent's tool-calling loop —
Max invokes this directly via `_execute_bash`, e.g.:

    python utils/google_drive.py --list
    python utils/google_drive.py --list --folder-id <id>
    python utils/google_drive.py --info <file_id>

Requires GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and GOOGLE_DRIVE_REFRESH_TOKEN
in secrets/.env (refresh token needs the drive.readonly scope at minimum).
"""

import argparse
import json
import os

import requests
from dotenv import load_dotenv

load_dotenv("secrets/.env")

TOKEN_URL = "https://oauth2.googleapis.com/token"
FILES_URL = "https://www.googleapis.com/drive/v3/files"
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
FIELDS = "id,name,mimeType,modifiedTime,size,parents,webViewLink"


def _get_access_token():
    response = requests.post(
        TOKEN_URL,
        data={
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
            "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
            "refresh_token": os.getenv("GOOGLE_DRIVE_REFRESH_TOKEN"),
            "grant_type": "refresh_token",
        },
    )
    response.raise_for_status()
    return response.json()["access_token"]


def list_files(folder_id=None):
    params = {
        "fields": f"files({FIELDS})",
        "pageSize": 100,
    }
    if folder_id:
        params["q"] = f"'{folder_id}' in parents and trashed = false"
    else:
        params["q"] = "trashed = false"
    response = requests.get(
        FILES_URL,
        headers={"Authorization": f"Bearer {_get_access_token()}"},
        params=params,
    )
    response.raise_for_status()
    files = response.json().get("files", [])
    for file in files:
        file["isFolder"] = file.get("mimeType") == FOLDER_MIME_TYPE
    return files


def get_file_info(file_id):
    response = requests.get(
        f"{FILES_URL}/{file_id}",
        headers={"Authorization": f"Bearer {_get_access_token()}"},
        params={"fields": FIELDS},
    )
    response.raise_for_status()
    info = response.json()
    info["isFolder"] = info.get("mimeType") == FOLDER_MIME_TYPE
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List files/folders")
    parser.add_argument("--folder-id", help="Restrict --list to this folder's contents")
    parser.add_argument("--info", metavar="FILE_ID", help="Get metadata for a single file/folder")
    args = parser.parse_args()

    if args.info:
        print(json.dumps(get_file_info(args.info), indent=2))
    elif args.list:
        print(json.dumps(list_files(args.folder_id), indent=2))
    else:
        parser.print_help()
