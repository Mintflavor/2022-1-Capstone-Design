FROM alpine
LABEL authors="mintflavor"
COPY . /app
RUN apk update && apk add --no-cache python3 py3-pip py3-numpy py3-pandas py3-flask && \
    pip3 install --no-cache-dir -r /app/requirements_alpine.txt

WORKDIR /app
EXPOSE 80

CMD ["python", "app.py"]