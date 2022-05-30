FROM apache/airflow:2.3.0

USER root

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         vim \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && pip install --progress-bar off -r requirements.txt
