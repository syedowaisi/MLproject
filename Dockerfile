FROM python:3.10-slim
WORKDIR /application
COPY . /application

# RUN apt update -y && apt install awscli -y
RUN apt-get update -y && apt-get install -y awscli

RUN pip install -r requirements.txt
CMD ["python","application.py"]