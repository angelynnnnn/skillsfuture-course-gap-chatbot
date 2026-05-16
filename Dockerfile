FROM python:3.11-slim
WORKDIR /app
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["sh", "-c", "python src/ingest_chroma.py && streamlit run app.py --server.address=0.0.0.0 --server.port=8501"]