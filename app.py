from flask import Flask, request , jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return "hey"

if __name__ == "__main__":
    app.run(debug=True)  