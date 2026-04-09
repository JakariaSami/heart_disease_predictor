FROM python:3.11-slim

WORKDIR /app

COPY requirements.prod.txt .
RUN pip install --no-cache-dir -r requirements.prod.txt

COPY src/ ./src/
COPY api/ ./api/

RUN mkdir -p models

COPY startup.sh .
RUN chmod +x startup.sh

ENV PYTHONPATH=/app

CMD ["./startup.sh"]