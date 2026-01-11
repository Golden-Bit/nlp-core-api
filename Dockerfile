FROM ubuntu:22.04

LABEL maintainer="tuo_nome@example.com"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Rome

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ca-certificates \
    tzdata \
    bash && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /build_app

# Migliora caching: prima requirements, poi il resto
COPY requirements.txt /build_app/requirements.txt
RUN python3.10 -m pip install --no-cache-dir --upgrade pip && \
    python3.10 -m pip install --no-cache-dir -r requirements.txt

COPY . /build_app

# Entry point per persistenza cartelle
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV APP_DIR=/build_app
ENV PERSIST_ROOT=/persist

EXPOSE 8100

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100", "--workers", "1"]
