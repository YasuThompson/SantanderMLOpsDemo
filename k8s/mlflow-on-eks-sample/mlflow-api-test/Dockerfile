FROM python:3.9.19-alpine3.18
WORKDIR /usr/src/app
COPY ./app .
RUN pip install pipenv
RUN pipenv install
EXPOSE 5000
CMD ["pipenv", "run", "python", "api.py"]