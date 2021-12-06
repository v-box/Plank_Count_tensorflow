FROM python:3.9

WORKDIR /plank

COPY ./requirements.txt /plank/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /plank/requirements.txt

COPY ./app /plank/app

CMD ["python", "app/main.py"]

