FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD model.joblib /code/model.joblib
ADD train.py /code/train.py
ADD run.py /code/run.py

ENTRYPOINT [ "python3", "/code/run.py" ]