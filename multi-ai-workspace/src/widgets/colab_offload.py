"""Colab Offload Widget - Upload and run code in Google Colab.

Enables offloading heavy compute tasks to Google Colab by uploading
notebooks/code to Google Drive and generating Colab execution links.
"""

import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import pickle
import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from ..utils.logger import get_logger

logger = get_logger(__name__)


# Google API scopes
SCOPES = [
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/drive'
]


class ColabOffload:
    """
    Colab Offload Widget.

    Provides functionality to:
    - Upload notebooks/files to Google Drive
    - Generate Colab links for execution
    - Monitor execution status
    - Retrieve results
    """

    def __init__(
        self,
        credentials_path: Optional[str] = None,
        token_path: Optional[str] = None,
        drive_folder_name: str = "MultiAI_Workspace"
    ):
        """
        Initialize Colab Offload.

        Args:
            credentials_path: Path to OAuth2 credentials.json
            token_path: Path to save/load token.pickle
            drive_folder_name: Name of Drive folder to use
        """
        self.credentials_path = credentials_path or "credentials.json"
        self.token_path = token_path or "token.pickle"
        self.drive_folder_name = drive_folder_name

        self.creds = None
        self.drive_service = None
        self.folder_id = None

        logger.info("ColabOffload initialized")

    def authenticate(self) -> bool:
        """
        Authenticate with Google Drive API.

        Returns:
            True if authenticated successfully
        """
        try:
            # Load existing token
            if os.path.exists(self.token_path):
                with open(self.token_path, 'rb') as token:
                    self.creds = pickle.load(token)

            # Refresh or get new token
            if not self.creds or not self.creds.valid:
                if self.creds and self.creds.expired and self.creds.refresh_token:
                    self.creds.refresh(Request())
                else:
                    if not os.path.exists(self.credentials_path):
                        logger.error(f"Credentials file not found: {self.credentials_path}")
                        return False

                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path,
                        SCOPES
                    )
                    self.creds = flow.run_local_server(port=0)

                # Save token
                with open(self.token_path, 'wb') as token:
                    pickle.dump(self.creds, token)

            # Build Drive service
            self.drive_service = build('drive', 'v3', credentials=self.creds)

            # Get or create workspace folder
            self.folder_id = self._get_or_create_folder()

            logger.info("Google Drive authentication successful")
            return True

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False

    def _get_or_create_folder(self) -> str:
        """
        Get or create the workspace folder in Drive.

        Returns:
            Folder ID
        """
        try:
            # Search for existing folder
            query = f"name='{self.drive_folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            response = self.drive_service.files().list(
                q=query,
                spaces='drive',
                fields='files(id, name)'
            ).execute()

            folders = response.get('files', [])

            if folders:
                return folders[0]['id']

            # Create folder if not exists
            folder_metadata = {
                'name': self.drive_folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }

            folder = self.drive_service.files().create(
                body=folder_metadata,
                fields='id'
            ).execute()

            logger.info(f"Created Drive folder: {self.drive_folder_name}")
            return folder['id']

        except Exception as e:
            logger.error(f"Folder creation failed: {e}")
            raise

    def upload_notebook(
        self,
        notebook_path: str | Path,
        custom_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload Jupyter notebook to Google Drive.

        Args:
            notebook_path: Path to .ipynb file
            custom_name: Custom filename (default: original name)

        Returns:
            Upload result with file ID and Colab link
        """
        if not self.drive_service:
            if not self.authenticate():
                return {"error": "Authentication failed"}

        notebook_path = Path(notebook_path)

        if not notebook_path.exists():
            return {"error": f"File not found: {notebook_path}"}

        if notebook_path.suffix != ".ipynb":
            return {"error": "File must be a .ipynb Jupyter notebook"}

        try:
            # Prepare file metadata
            file_name = custom_name or notebook_path.name
            file_metadata = {
                'name': file_name,
                'parents': [self.folder_id]
            }

            # Upload file
            media = MediaFileUpload(
                str(notebook_path),
                mimetype='application/json',
                resumable=True
            )

            file = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, name, webViewLink'
            ).execute()

            file_id = file['id']

            # Generate Colab link
            colab_link = f"https://colab.research.google.com/drive/{file_id}"

            logger.info(f"Uploaded notebook: {file_name} (ID: {file_id})")

            return {
                "success": True,
                "file_id": file_id,
                "file_name": file_name,
                "colab_link": colab_link,
                "drive_link": file.get('webViewLink'),
                "folder": self.drive_folder_name
            }

        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return {"error": str(e)}

    def upload_code_as_notebook(
        self,
        code: str,
        filename: str = "generated_notebook.ipynb",
        language: str = "python"
    ) -> Dict[str, Any]:
        """
        Upload code as a Jupyter notebook.

        Args:
            code: Code content
            filename: Notebook filename
            language: Code language (python, r, etc.)

        Returns:
            Upload result
        """
        if not self.drive_service:
            if not self.authenticate():
                return {"error": "Authentication failed"}

        try:
            # Create notebook structure
            notebook = {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": code.split("\n")
                    }
                ],
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3"
                    },
                    "language_info": {
                        "name": language
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 4
            }

            # Save temporarily
            temp_path = Path(f"/tmp/{filename}")
            with open(temp_path, 'w') as f:
                json.dump(notebook, f, indent=2)

            # Upload
            result = self.upload_notebook(temp_path, filename)

            # Clean up
            temp_path.unlink()

            return result

        except Exception as e:
            logger.error(f"Code upload failed: {e}")
            return {"error": str(e)}

    def list_uploaded_files(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        List uploaded files in workspace folder.

        Args:
            limit: Maximum files to return

        Returns:
            List of file metadata
        """
        if not self.drive_service:
            if not self.authenticate():
                return []

        try:
            query = f"'{self.folder_id}' in parents and trashed=false"
            response = self.drive_service.files().list(
                q=query,
                pageSize=limit,
                fields='files(id, name, createdTime, modifiedTime, webViewLink)',
                orderBy='modifiedTime desc'
            ).execute()

            files = response.get('files', [])

            return [
                {
                    "id": f['id'],
                    "name": f['name'],
                    "created": f.get('createdTime'),
                    "modified": f.get('modifiedTime'),
                    "drive_link": f.get('webViewLink'),
                    "colab_link": f"https://colab.research.google.com/drive/{f['id']}"
                    if f['name'].endswith('.ipynb') else None
                }
                for f in files
            ]

        except Exception as e:
            logger.error(f"File listing failed: {e}")
            return []

    def delete_file(self, file_id: str) -> Dict[str, Any]:
        """
        Delete a file from Drive.

        Args:
            file_id: Google Drive file ID

        Returns:
            Delete result
        """
        if not self.drive_service:
            if not self.authenticate():
                return {"error": "Authentication failed"}

        try:
            self.drive_service.files().delete(fileId=file_id).execute()

            logger.info(f"Deleted file: {file_id}")

            return {
                "success": True,
                "file_id": file_id
            }

        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return {"error": str(e)}

    def get_offload_info(self) -> Dict[str, Any]:
        """
        Get information about Colab Offload setup.

        Returns:
            Setup information
        """
        is_authenticated = self.creds is not None and self.creds.valid

        return {
            "authenticated": is_authenticated,
            "credentials_path": self.credentials_path,
            "drive_folder": self.drive_folder_name,
            "folder_id": self.folder_id,
            "instructions": {
                "1": "Get OAuth2 credentials from https://console.cloud.google.com/",
                "2": "Enable Google Drive API",
                "3": "Download credentials.json",
                "4": "Place in workspace root or specify path",
                "5": "Call authenticate() to get token"
            }
        }

    def generate_quickstart_notebook(
        self,
        code_snippet: str,
        setup_commands: Optional[List[str]] = None
    ) -> str:
        """
        Generate a Colab-ready notebook with setup.

        Args:
            code_snippet: Main code to run
            setup_commands: List of setup commands (pip install, etc.)

        Returns:
            Notebook JSON string
        """
        cells = []

        # Setup cell
        if setup_commands:
            setup_code = "\n".join(f"!{cmd}" for cmd in setup_commands)
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": setup_code.split("\n")
            })

        # Main code cell
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code_snippet.split("\n")
        })

        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }

        return json.dumps(notebook, indent=2)
