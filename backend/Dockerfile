# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3-slim

# EXPOSE 8000
# EXPOSE 5678
# EXPOSE 11434

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

COPY . .

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser
WORKDIR /app/src
# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "main.py"]
