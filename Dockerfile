FROM python:3.8-slim-buster
RUN apt update -y  &&  apt upgrade -y && apt-get update 
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8080
CMD ["python","app.py"]
# CMD ["/usr/local/bin/python","src/components/data_ingestion.py" ]