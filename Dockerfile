FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# install system fonts and utilities before Python deps
RUN apt-get update \
    && apt-get install -y fontconfig \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/share/fonts/truetype/malgun
COPY malgun.ttf malgunbd.ttf /usr/share/fonts/truetype/malgun/
RUN fc-cache -f -v

# install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# copy notebooks and credentials that should live inside the container
COPY db_info.txt final.ipynb run.ipynb api_key.txt ./

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
