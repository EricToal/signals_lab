FROM python:3.12-slim

# Install git
RUN apt-get update && \ 
    apt-get upgrade -y && \
    apt-get install git -y && \
    apt-get clean

RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip

# Useful environment variables for python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1