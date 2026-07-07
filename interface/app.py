from flask import Flask, Blueprint, render_template

from interface import pages
from interface import api

if __name__ == '__main__':

    app = Flask(__name__)
    app.register_blueprint(pages.bp)
    app.register_blueprint(api.bp)
    
    # Bind explicitly to your Tailscale IP on port 5000
    app.run(host='100.95.135.118', port=5000, debug=True)