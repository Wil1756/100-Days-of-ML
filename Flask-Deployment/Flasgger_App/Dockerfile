FROM python:3.12.4 

COPY . /usr/ML/app

EXPOSE 5000

WORKDIR /usr/ML/app

RUN pip install -r requirements.txt

CMD ["python", "app.py"]