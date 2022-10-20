from flask import Flask

app = Flask(__name__)


@app.route('/')
def flask_mongodb_atlas():
    return "flask mongodb atlas!"

if __name__ == '__main__':
    app.run(port=5002)