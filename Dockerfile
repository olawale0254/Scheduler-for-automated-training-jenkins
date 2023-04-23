FROM python:3.8

WORKDIR /app
# CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--reload"]

COPY src/* ./
RUN pip install --no-cache-dir -r ./requirements.txt
