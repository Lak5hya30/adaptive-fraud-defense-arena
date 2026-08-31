FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "streamlit run app/Home.py --server.address=0.0.0.0 --server.port=${PORT} --server.headless=true"]