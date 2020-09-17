FROM tensorflow:latest

WORKDIR /opt/recsyslib

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    unixodbc-dev \
    python3-pip \
    python3-dev \
    python-psycopg2 \
    libpq-dev \
    postgresql-server-dev-all \
    vim

RUN pip3 install --upgrade \
    pip \
    setuptools

COPY . .

RUN pip3 install -r requirements.txt && \
    make install

CMD ["python3", "recsyslib/recsyslib.py"]
