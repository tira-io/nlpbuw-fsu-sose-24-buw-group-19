# docker build -t fschlatt/authorship-verification-trivial:0.0.1 .
FROM fschlatt/natural-language-processing-exercises:0.0.1

RUN pip3 install sklearn-crfsuite 

ADD run.py /code/run.py

ENTRYPOINT [ "python3", "/code/run.py" ]
