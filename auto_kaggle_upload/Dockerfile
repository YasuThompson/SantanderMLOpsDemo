FROM python:3.7.7-buster as builder

# Install python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM python:3.7.7-slim-buster as runner
COPY --from=builder /usr/local/lib/python3.7/site-packages /usr/local/lib/python3.7/site-packages
# COPY --from=builder /usr/local/bin /usr/local/bin # <-いる？？

# Install linux packages # <-いる？？
# RUN apt-get update && apt-get install -y git unzip

# WORKDIR /work # <-docker composeで指定するのでいらない．
