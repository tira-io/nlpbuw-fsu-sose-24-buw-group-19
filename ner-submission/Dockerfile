FROM fschlatt/natural-language-processing-exercises:0.0.1

RUN pip3 install sklearn_crfsuite
RUN pip3 install spacy

ADD ner_model.joblib /code/ner_model.joblib
ADD train.py /code/train.py
ADD run.py /code/run.py

WORKDIR /code

ENTRYPOINT [ "python3", "/code/run.py" ]
