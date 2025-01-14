FROM python:3.12-slim

WORKDIR /workspace

# Install git and other useful packages
RUN apt-get update && \ 
    apt-get install git -y && \
    apt-get install gcc -y && \
    apt-get install g++ -y && \
    apt-get clean

RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt
COPY . .

# Useful environment variables for python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1