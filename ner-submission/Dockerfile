# docker build -t fschlatt/authorship-verification-trivial:0.0.1 .
FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD ner_model.joblib /code/ner_model.joblib
ADD train.py /code/train.py
ADD run.py /code/run.py

ENTRYPOINT [ "python3", "/code/run.py" ]
