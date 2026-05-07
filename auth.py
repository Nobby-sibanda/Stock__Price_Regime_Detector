"""
Authentication Blueprint: /register, /login, /logout.
Uses werkzeug password hashing; sessions managed by Flask-Login.
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, UserMixin, LoginManager

from werkzeug.security import generate_password_hash, check_password_hash

from . import db as _db

auth_bp = Blueprint("auth", __name__)


# ── Flask-Login user class ────────────────────────────────────────────────────

class User(UserMixin):
    def __init__(self, row):
        self.id       = row["id"]
        self.username = row["username"]

    def get_id(self):
        return str(self.id)


def init_login_manager(app):
    lm = LoginManager(app)
    lm.login_view        = "auth.login"
    lm.login_message     = "Please log in to run an analysis."
    lm.login_message_category = "info"

    @lm.user_loader
    def load_user(user_id):
        row = _db.get_user_by_id(int(user_id))
        return User(row) if row else None

    return lm


# ── Routes ────────────────────────────────────────────────────────────────────

@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm", "")

        error = None
        if not username:
            error = "Username is required."
        elif len(username) < 3:
            error = "Username must be at least 3 characters."
        elif not password:
            error = "Password is required."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != confirm:
            error = "Passwords do not match."
        elif _db.get_user_by_username(username):
            error = f"Username '{username}' is already taken."

        if error:
            return render_template("register.html", error=error, username=username)

        pw_hash = generate_password_hash(password)
        user_id = _db.create_user(username, pw_hash)
        row     = _db.get_user_by_id(user_id)
        login_user(User(row))
        return redirect(url_for("index"))

    return render_template("register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        row = _db.get_user_by_username(username)
        if not row or not check_password_hash(row["password_hash"], password):
            return render_template("login.html", error="Invalid username or password.",
                                   username=username)

        login_user(User(row), remember=request.form.get("remember") == "on")
        next_page = request.args.get("next") or url_for("index")
        return redirect(next_page)

    return render_template("login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))
