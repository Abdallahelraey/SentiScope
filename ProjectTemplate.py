import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

template_config = {
    "project_name": "SentiScope",
    "files": [
        ".github/workflows/.gitkeep",
        "src/{project_name}/__init__.py",
        "src/{project_name}/components/__init__.py",
        "src/{project_name}/utils/__init__.py",
        "src/{project_name}/utils/common.py",
        "src/{project_name}/logging/__init__.py",
        "src/{project_name}/config/__init__.py",
        "src/{project_name}/config/configuration.py",
        "src/{project_name}/pipeline/__init__.py",
        "src/{project_name}/entity/__init__.py",
        "src/{project_name}/constants/__init__.py",
        "config/config.yaml",
        "params.yaml",
        "app.py",
        "README.md",
        "LICENSE",
        ".env",
        ".env.example",
        ".gitignore",
        "main.py",
        "Dockerfile",
        "requirements.txt",
        "setup.py",
        "poc/POC.ipynb"
    ]
}

def create_project_structure(template):
    project_name = template["project_name"]
    list_of_files = [file.format(project_name=project_name) for file in template["files"]]

    for filepath in list_of_files:
        filepath = Path(filepath)
        filedir, filename = os.path.split(filepath)

        if filedir != "":
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Creating directory: {filedir} for the file {filename}")

        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            with open(filepath, 'w') as f:
                pass
            logging.info(f"Creating empty file: {filepath}")
        else:
            logging.info(f"{filename} already exists")


if __name__ == "__main__":
    create_project_structure(template_config)
