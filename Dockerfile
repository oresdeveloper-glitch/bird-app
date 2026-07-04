FROM tensorflow/tensorflow:2.13.0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p static/uploads dataset

ENV PORT=7860
ENV FLASK_DEBUG=0

EXPOSE 7860

CMD ["python", "app.py", "--no-ssl"]
