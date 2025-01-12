from flask import Flask

app = Flask(__name__)

# Import routes after the app is created to avoid circular imports
from app.routes import api_calls, home
