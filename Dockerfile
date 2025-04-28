FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["fastapi", "run", "./request_handler.py", "--host", "0.0.0.0", "--port", "8000", "--reload"]
