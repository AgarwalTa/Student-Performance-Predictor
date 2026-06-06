# app.py
import os
import json
import tempfile
import logging
from functools import wraps
from datetime import datetime, timezone, timedelta

from flask import (
    Flask, request, render_template, redirect, url_for, session, flash, jsonify,
    send_from_directory
)
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash

# --- Local pipeline import ---
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.utils.generate_user_reports import generate_user_reports  # used only if admin enables it

# --- Paths & Artifacts ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
USERS_FILE = os.path.join(ARTIFACTS_DIR, "users.csv")
PREDICTIONS_FILE = os.path.join(ARTIFACTS_DIR, "predictions.csv")
SETTINGS_FILE = os.path.join(ARTIFACTS_DIR, "app_settings.json")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# --- Logging ---
logging.basicConfig(filename=os.path.join(BASE_DIR, "app.log"),
                    level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# --- Flask app ---
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecretkey123")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.permanent_session_lifetime = timedelta(hours=8)

@app.context_processor
def inject_now():
    return {"now": datetime.utcnow}

# ------------------------------
# --- SETTINGS (persistent) ---
# ------------------------------
DEFAULT_SETTINGS = {
    "MAX_PREDICTIONS_PER_USER": 3,
    "AUTO_GENERATE_REPORTS": False,
    "THEME": "light",
    "NOTIFICATIONS": True
}

def load_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
                # ensure defaults exist
                for k, v in DEFAULT_SETTINGS.items():
                    d.setdefault(k, v)
                return d
    except Exception:
        logging.exception("Failed to load settings, using defaults.")
    return DEFAULT_SETTINGS.copy()

def save_settings(s):
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, dir=ARTIFACTS_DIR, encoding="utf-8") as tf:
            json.dump(s, tf, indent=2)
            tmpname = tf.name
        os.replace(tmpname, SETTINGS_FILE)
    except Exception:
        logging.exception("Failed to save settings.")

SETTINGS = load_settings()

# ------------------------------
# --- File utilities (atomic writes) ---
# ------------------------------
def atomic_csv_write(df: pd.DataFrame, path: str):
    dirpath = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dirpath)
    try:
        os.close(fd)
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass

def sanitize_csv_cell(val: str) -> str:
    if isinstance(val, str) and val and val[0] in ("=", "+", "-", "@"):
        return "'" + val
    return val

# ------------------------------
# --- Users load/save (hashes stored, not exposed) ---
# ------------------------------
def load_users():
    if os.path.exists(USERS_FILE) and os.path.getsize(USERS_FILE) > 0:
        df = pd.read_csv(USERS_FILE)
        users = {}
        for _, row in df.iterrows():
            users[row["username"]] = {
                "password": row["password"],
                "role": row.get("role", "user"),
                "status": row.get("status", "active")
            }
        # ensure admin exists
        if "admin" not in users:
            users["admin"] = {"password": generate_password_hash("admin123"), "role": "admin", "status": "active"}
            save_users(users)
        return users
    # no file -> create admin only
    return {"admin": {"password": generate_password_hash("admin123"), "role": "admin", "status": "active"}}

def save_users(users: dict):
    df = pd.DataFrame([
        {
            "username": sanitize_csv_cell(u),
            "password": d["password"],
            "role": d.get("role", "user"),
            "status": d.get("status", "active")
        }
        for u, d in users.items()
    ])
    atomic_csv_write(df, USERS_FILE)

# init users
USERS = load_users()

# ------------------------------
# --- Helpers: validation & safe predict ---
# ------------------------------
def to_float_clamped(value, minimum=0.0, maximum=None, name="value"):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid numeric value for {name}.")
    if maximum is not None:
        v = max(minimum, min(v, maximum))
    else:
        v = max(minimum, v)
    return v

def safe_predict(pipeline: PredictPipeline, features_df: pd.DataFrame):
    """
    Call pipeline.predict and coerce to float. Raise RuntimeError on failure.
    """
    try:
        out = pipeline.predict(features_df)
        # handle different return types
        if hasattr(out, "__len__") and not isinstance(out, (float, int)):
            val = float(out[0])
        else:
            val = float(out)
        if not (val == val):  # NaN check
            raise ValueError("Prediction returned NaN")
        return val
    except Exception:
        logging.exception("Model prediction failed")
        raise RuntimeError("Prediction engine error")

def sanity_correction(score: float, inputs: dict) -> float:
    """
    Apply conservative post-model sanity caps based on low inputs.
    This is intentionally conservative to avoid breaking model behavior.
    """
    study_hours = inputs.get("study_hours", 0)
    sleep_hours = inputs.get("sleep_hours", 0)
    attendance = inputs.get("attendance", 0)
    prev_grades = inputs.get("prev_grades", 0)

    # If many core inputs are very low -> cap aggressively
    low_inputs = sum([
        study_hours < 1,
        attendance < 40,
        sleep_hours < 4,
        prev_grades < 40
    ])

    # Severe low conditions -> strong cap
    if low_inputs >= 3:
        score = min(score, 60.0)

    # Specific common case: almost no study + low attendance
    if study_hours < 1 and attendance < 50:
        score = min(score, 55.0)

    # If overall average of core signals is very small -> cap
    core_avg = (study_hours + sleep_hours + attendance) / 3.0
    if core_avg < 30:
        score = min(score, 50.0)

    # Final generic sanity cap: do not artificially push weak inputs into top tier
    return max(0.0, min(score, 100.0))

# ------------------------------
# --- Save Prediction (app-level) ---
# ------------------------------
def save_prediction_app(username: str,
                        study_hours: float,
                        sleep_hours: float,
                        attendance: float,
                        predicted_score: float,
                        prev_grades: float = 0.0,
                        participation: float = 0.0,
                        assignments_completed: float = 0.0,
                        revision_hours: float = 0.0,
                        extracurricular_hours: float = 0.0):
    cols = ["username", "study_hours", "sleep_hours", "attendance",
            "prev_grades", "participation", "assignments_completed",
            "revision_hours", "extracurricular_hours",
            "predicted_score", "timestamp"]

    if os.path.exists(PREDICTIONS_FILE) and os.path.getsize(PREDICTIONS_FILE) > 0:
        df = pd.read_csv(PREDICTIONS_FILE)
    else:
        df = pd.DataFrame(columns=cols)

    new_row = {
        "username": username,
        "study_hours": study_hours,
        "sleep_hours": sleep_hours,
        "attendance": attendance,
        "prev_grades": prev_grades,
        "participation": participation,
        "assignments_completed": assignments_completed,
        "revision_hours": revision_hours,
        "extracurricular_hours": extracurricular_hours,
        "predicted_score": predicted_score,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    atomic_csv_write(df, PREDICTIONS_FILE)

    # Optionally generate report for this user (only if admin enabled)
    try:
        if SETTINGS.get("AUTO_GENERATE_REPORTS"):
            generate_user_reports(specific_user=username)
            logging.info("Auto-generated report for user: %s", username)
    except Exception:
        logging.exception("Report generation failed for user: %s", username)

# ------------------------------
# --- Auth decorator & helpers ---
# ------------------------------
def login_required(role=None):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            if "username" not in session:
                flash("Please login to continue.", "warning")
                return redirect(url_for("login"))
            if role and session.get("role") != role:
                flash("You do not have permission to access that page.", "danger")
                return redirect(url_for("home"))
            return f(*args, **kwargs)
        return wrapped
    return decorator

# ------------------------------
# --- Routes ---
# ------------------------------
@app.route("/")
def home():
    if "username" in session:
        if session.get("role") == "admin":
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("user_dashboard"))
    return render_template("home.html")

# --- Login / Logout / Register ---
@app.route("/login", methods=["GET", "POST"])
def login():
    global USERS
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        USERS = load_users()
        user = USERS.get(username or "")
        if user and check_password_hash(user["password"], password):
            if user.get("status") == "suspended":
                return render_template("login.html", result="Your account is suspended. Contact admin.")
            session.permanent = True
            session["username"] = username
            session["role"] = user["role"]
            if user["role"] == "admin":
                return redirect(url_for("admin_dashboard"))
            return redirect(url_for("user_dashboard"))
        return render_template("login.html", result="Invalid username or password")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    global USERS
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        if not username or not password:
            flash("Please provide username and password.", "warning")
            return redirect(url_for("register"))
        if password != confirm:
            flash("Passwords do not match.", "warning")
            return redirect(url_for("register"))
        USERS = load_users()
        if username in USERS:
            flash("Username already exists!", "danger")
            return redirect(url_for("register"))
        # minimal password policy: length >= 6
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "warning")
            return redirect(url_for("register"))

        hashed = generate_password_hash(password)
        USERS[username] = {"password": hashed, "role": "user", "status": "active"}
        save_users(USERS)
        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

# --- User Dashboard ---
@app.route("/user_dashboard")
@login_required(role="user")
def user_dashboard():
    username = session["username"]

    # Load user's prediction history
    if os.path.exists(PREDICTIONS_FILE) and os.path.getsize(PREDICTIONS_FILE) > 0:
        df = pd.read_csv(PREDICTIONS_FILE)
        user_predictions = df[df["username"] == username].copy()
    else:
        user_predictions = pd.DataFrame(columns=[
            "study_hours", "sleep_hours", "attendance", "prev_grades",
            "participation", "assignments_completed", "revision_hours",
            "extracurricular_hours", "predicted_score", "timestamp"
        ])

    # --- Limit predictions ---
    MAX_PREDICTIONS_PER_USER = SETTINGS.get("MAX_PREDICTIONS_PER_USER", 3)
    remaining_predictions = max(0, MAX_PREDICTIONS_PER_USER - len(user_predictions))

    return render_template(
        "user_dashboard.html",
        username=username,
        history=user_predictions.to_html(index=False, classes="table table-striped", escape=False),
        chart_data=user_predictions.to_dict(orient="records"),
        remaining_predictions=remaining_predictions
    )

@app.route("/chart_data")
@login_required()
def chart_data():
    username = session["username"]
    if os.path.exists(PREDICTIONS_FILE) and os.path.getsize(PREDICTIONS_FILE) > 0:
        df = pd.read_csv(PREDICTIONS_FILE)
        user_df = df[df["username"] == username].sort_values("timestamp")
        records = user_df[["timestamp", "study_hours", "sleep_hours", "attendance",
                           "prev_grades","participation","assignments_completed",
                           "revision_hours","extracurricular_hours","predicted_score"]].to_dict(orient="records")
    else:
        records = []
    return jsonify(records)

# --- Admin Dashboard ---
@app.route("/admin_dashboard")
@login_required(role="admin")
def admin_dashboard():
    if os.path.exists(PREDICTIONS_FILE) and os.path.getsize(PREDICTIONS_FILE) > 0:
        df = pd.read_csv(PREDICTIONS_FILE)
    else:
        df = pd.DataFrame(columns=[
            "username","study_hours","sleep_hours","attendance","prev_grades","participation",
            "assignments_completed","revision_hours","extracurricular_hours","predicted_score","timestamp"
        ])

    # Build sanitized user view (do not expose password hashes)
    users_full = load_users()
    users_safe = {u: {"role": d.get("role"), "status": d.get("status")} for u, d in users_full.items()}

    all_predictions = {user: df[df["username"] == user].to_dict(orient="records") for user in users_full.keys()}

    logs = []
    if os.path.exists(os.path.join(BASE_DIR, "app.log")):
        try:
            with open(os.path.join(BASE_DIR, "app.log"), "r", encoding="utf-8", errors="ignore") as f:
                logs = f.readlines()[-200:]
        except Exception:
            logs = ["Could not read logs."]
    return render_template("admin_dashboard.html",
                           users=users_safe,
                           all_predictions=all_predictions,
                           logs=logs,
                           settings=SETTINGS)

@app.route("/predict", methods=["POST"])
@login_required()
def predict():
    try:
        username = session.get("username", "guest")

        # --- Load user's prediction history ---
        if os.path.exists(PREDICTIONS_FILE) and os.path.getsize(PREDICTIONS_FILE) > 0:
            df = pd.read_csv(PREDICTIONS_FILE)
            user_predictions = df[df["username"] == username]
        else:
            user_predictions = pd.DataFrame()

        # --- Enforce prediction limit ---
        MAX_PREDICTIONS_PER_USER = SETTINGS.get("MAX_PREDICTIONS_PER_USER", 3)
        if len(user_predictions) >= MAX_PREDICTIONS_PER_USER:
            flash(f"You have reached the maximum of {MAX_PREDICTIONS_PER_USER} predictions.", "warning")
            return redirect(url_for("user_dashboard"))

        # --- Collect input with validation & clamping ---
        try:
            study_hours = to_float_clamped(request.form.get("study_hours", 0), 0, 24, "study_hours")
            sleep_hours = to_float_clamped(request.form.get("sleep_hours", 0), 0, 24, "sleep_hours")
            attendance = to_float_clamped(request.form.get("attendance", 0), 0, 100, "attendance")
            prev_grades = to_float_clamped(request.form.get("prev_grades", 0), 0, 100, "prev_grades")
            participation = to_float_clamped(request.form.get("participation", 0), 0, 100, "participation")
            assignments_completed = to_float_clamped(request.form.get("assignments_completed", 0), 0, 100, "assignments_completed")
            revision_hours = to_float_clamped(request.form.get("revision_hours", 0), 0, 100, "revision_hours")
            extracurricular_hours = to_float_clamped(request.form.get("extracurricular_hours", 0), 0, 100, "extracurricular_hours")
        except ValueError as err:
            flash(str(err), "warning")
            return redirect(url_for("user_dashboard"))

        # --- Prepare Data ---
        data = CustomData(
            study_hours, sleep_hours, attendance,
            prev_grades, participation, assignments_completed,
            revision_hours, extracurricular_hours
        )
        df_input = data.get_data_as_dataframe()
        pipeline = PredictPipeline()

        # --- Safe prediction ---
        base_score = safe_predict(pipeline, df_input)

        # --- Cap predicted score at 100 & non-negative ---
        base_score = max(0.0, min(base_score, 100.0))

        # --- Sanity correction layer (prevent unrealistic high predictions on weak inputs) ---
        base_score = sanity_correction(base_score, {
            "study_hours": study_hours,
            "sleep_hours": sleep_hours,
            "attendance": attendance,
            "prev_grades": prev_grades
        })

        if base_score >= 100:
            flash("⚠️ Predicted score capped at 100%", "warning")

        # --- Save Prediction ---
        save_prediction_app(
            username, study_hours, sleep_hours, attendance, base_score,
            prev_grades, participation, assignments_completed,
            revision_hours, extracurricular_hours
        )

        # --- Generate Recommendations ---
        recommendations = []

        # Core parameters
        pipeline_for_delta = PredictPipeline()
        try:
            for col, add, emoji in [
                ("study_hours", 1, "💡"),
                ("sleep_hours", 1, "💤"),
                ("attendance", 5, "📅")
            ]:
                df_mod = df_input.copy()
                # handle attendance special cap
                if col == "attendance":
                    df_mod[col] = min(df_mod[col].iloc[0] + add, 100)
                else:
                    df_mod[col] = df_mod[col].iloc[0] + add
                delta_pred = safe_predict(pipeline_for_delta, pd.DataFrame([df_mod.iloc[0]]))
                delta_raw = delta_pred - base_score
                potential_score = min(base_score + max(0.0, delta_raw), 100.0)
                delta = potential_score - base_score
                if delta > 0:
                    recommendations.append(f"{emoji} {col.replace('_',' ').title()} +{add} → +{delta:.1f} points")
        except Exception:
            logging.exception("Delta computation failed for core params")

        # Optional parameters
        optional_params = [
            ("prev_grades", 5, "📘"),
            ("participation", 1, "🗣️"),
            ("assignments_completed", 1, "📝"),
            ("revision_hours", 1, "📚"),
            ("extracurricular_hours", 1, "⚽")
        ]
        try:
            for col, add, emoji in optional_params:
                df_mod = df_input.copy()
                df_mod[col] = df_mod[col].iloc[0] + add
                delta_pred = safe_predict(pipeline_for_delta, pd.DataFrame([df_mod.iloc[0]]))
                delta_raw = delta_pred - base_score
                potential_score = min(base_score + max(0.0, delta_raw), 100.0)
                delta = potential_score - base_score
                if delta > 0:
                    recommendations.append(f"{emoji} Increase {col.replace('_',' ')} by {add} → +{delta:.1f} points")
        except Exception:
            logging.exception("Delta computation failed for optional params")

        # Score feedback
        if base_score >= 85:
            recommendations.append("🏆 Excellent! Keep up the good work.")
        elif base_score >= 70:
            recommendations.append("👍 Good! A little more effort can boost your score.")
        else:
            recommendations.append("⚠️ Focus more on study, sleep, and attendance to improve.")

        for r in recommendations:
            logging.info("Recommendation for %s: %s", username, r)

        flash(f"Predicted Score: {base_score:.2f}", "info")
        flash(" ".join(recommendations), "success")
        return redirect(url_for("user_dashboard"))

    except Exception as e:
        logging.exception("Prediction error")
        flash(f"Error during prediction: {str(e)}", "danger")
        return redirect(url_for("user_dashboard"))

# --- Admin User Management ---
@app.route("/delete_user/<username>", methods=["POST"])
@login_required(role="admin")
def delete_user(username):
    users = load_users()
    if username in users and users[username].get("role") != "admin":
        users.pop(username)
        save_users(users)
        flash(f"User {username} deleted.", "info")
    else:
        flash("Cannot delete that user.", "warning")
    return redirect(url_for("admin_dashboard"))

@app.route("/suspend_user/<username>", methods=["POST"])
@login_required(role="admin")
def suspend_user(username):
    users = load_users()
    if username in users and users[username].get("role") != "admin":
        users[username]["status"] = "suspended"
        save_users(users)
        flash(f"User {username} suspended.", "info")
    return redirect(url_for("admin_dashboard"))

@app.route("/reactivate_user/<username>", methods=["POST"])
@login_required(role="admin")
def reactivate_user(username):
    users = load_users()
    if username in users and users[username].get("role") != "admin":
        users[username]["status"] = "active"
        save_users(users)
        flash(f"User {username} reactivated.", "info")
    return redirect(url_for("admin_dashboard"))

# --- Admin Settings ---
@app.route("/settings", methods=["POST"])
@login_required(role="admin")
def settings():
    try:
        # Pull site settings from form; fall back to current settings
        theme = request.form.get("theme", SETTINGS.get("THEME", "light"))
        notifications = bool(request.form.get("notifications")) if "notifications" in request.form else SETTINGS.get("NOTIFICATIONS", True)
        max_predictions = int(request.form.get("max_predictions", SETTINGS.get("MAX_PREDICTIONS_PER_USER", 3)))
        auto_reports = bool(request.form.get("auto_generate_reports")) if "auto_generate_reports" in request.form else SETTINGS.get("AUTO_GENERATE_REPORTS", False)

        SETTINGS["THEME"] = theme
        SETTINGS["NOTIFICATIONS"] = notifications
        SETTINGS["MAX_PREDICTIONS_PER_USER"] = max_predictions
        SETTINGS["AUTO_GENERATE_REPORTS"] = auto_reports

        save_settings(SETTINGS)
        logging.info("Settings updated: %s", SETTINGS)
        flash("Settings updated.", "success")
    except Exception:
        logging.exception("Settings error")
        flash("Could not update settings.", "danger")
    return redirect(url_for("admin_dashboard"))

# --- Forgot / Reset Password (Option A: disabled) ---
@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        # Do not provide insecure reset. Ask user to contact admin.
        flash("Password reset via this portal is disabled. Please contact admin to reset your password.", "warning")
        return redirect(url_for("login"))
    return render_template("forgot_password.html")

@app.route("/reset_password", methods=["GET", "POST"])
def reset_password():
    # Reset flow disabled for security (Option A). Use admin intervention.
    flash("Password reset via this portal is disabled. Please contact admin.", "warning")
    return redirect(url_for("login"))

@app.route('/reports/<path:filename>')
def serve_reports(filename):
    return send_from_directory(ARTIFACTS_DIR, filename)

@app.route("/generate_reports", methods=["POST"])
@login_required(role="admin")
def generate_reports():
    try:
        generate_user_reports()
        flash("✅ Power BI reports regenerated successfully!", "success")
    except Exception as e:
        logging.exception("Report generation failed")
        flash(f"⚠️ Report generation failed: {e}", "danger")
    return redirect(url_for("admin_dashboard"))

# --- Run App ---
if __name__ == "__main__":
    app.run(debug=True)
