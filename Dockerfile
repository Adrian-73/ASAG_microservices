FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

COPY ./requirements.txt ./requirements.txt

RUN pip3 install torch==2.2.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install --no-cache-dir --upgrade -r ./requirements.txt

COPY . .

RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader stopwords

EXPOSE  8000

CMD ["fastapi", "run", "main.py", "--port", "8000"]