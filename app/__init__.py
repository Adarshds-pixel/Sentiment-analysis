import logging
from flask import Flask
from flask_cors import CORS


def create_app():
    app = Flask(__name__, static_folder='static', template_folder='templates')

    # Enable CORS
    CORS(app)

    # Logging setup
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    handler.setLevel(logging.INFO)

    app.logger.handlers = []
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)

    # Register routes
    from app.routes import register_routes
    register_routes(app)

    return app