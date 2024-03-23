FROM python:3.11-buster as builder
#FROM ubuntu:22.04 
#ENV PROJECT=santander_img

#ENV PYTHON_VERSION=3.8
#COPY requirements.txt .
#RUN pip install -r requirements.txt

RUN apt update 
# RUN apt install python3-pip

# RUN apt install -y python${PYTHON_VERSION} python3-pip 
# RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python & \
#     ln -s /usr/bin/pip3 /usr/bin/pip

FROM python:3.11-buster as runner
COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages

RUN pip install --upgrade pip

RUN pip install numpy pandas xgboost

