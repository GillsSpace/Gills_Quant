from flask import Flask, Blueprint, render_template
from datetime import datetime, timedelta, timezone

from logic.lib_clients import *

bp = Blueprint('main', __name__)

@bp.route('/')
def home():
    sd_client = create_client_schwab()
    rt_delta = timedelta(seconds=sd_client.tokens._refresh_token_timeout) - (datetime.now(timezone.utc) - sd_client.tokens._refresh_token_issued)
    return render_template('pages/home.html', tokens=f"Refresh token expires in: {str(rt_delta)[:-13]} hours")

