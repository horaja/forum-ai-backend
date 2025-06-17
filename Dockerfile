FROM python:3.9-slim

WORKDIR /usr/src/app

ENV FLASK_APP app/main.py

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /usr/src/app/app

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]