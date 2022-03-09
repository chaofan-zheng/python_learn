import time

from test_app import app


@app.task
def add(x, y):
    time.sleep(60)
    return x + y
