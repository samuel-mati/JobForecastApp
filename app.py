from flask import Flask, render_template, jsonify, redirect, url_for, request
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
import pickle
import pandas as pd

app = Flask(__name__)
app.secret_key = "techpulse-ea-secret-2026"  # TODO: move to env var in production

DB_PATH = "users.db"


# ── Database ──────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                name     TEXT    NOT NULL,
                email    TEXT    NOT NULL UNIQUE,
                password TEXT    NOT NULL,
                created  DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

init_db()


# ── Auth ──────────────────────────────────────────────────────────────────────

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "index"

class User(UserMixin):
    def __init__(self, id, name, email):
        self.id    = id
        self.name  = name
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return User(row["id"], row["name"], row["email"]) if row else None

@login_manager.unauthorized_handler
def unauthorized():
    return redirect(url_for("index") + "#login")


# ── Data loading ──────────────────────────────────────────────────────────────

def _parse_salary_midpoint(val):
    if pd.isna(val):
        return float("nan")
    try:
        parts = str(val).strip().split("-")
        return (float(parts[0]) + float(parts[1])) / 2
    except (ValueError, IndexError):
        return float("nan")

df = pd.read_csv("data/cleaned_jobs_dataset.csv")
df["salary"] = df["salary_range_usd"].apply(_parse_salary_midpoint)
df["year"]   = pd.to_datetime(df["posting_date"], errors="coerce").dt.year

demand_df = pd.read_csv("data/job_demand.csv")
demand_df.columns    = demand_df.columns.str.strip().str.lower()
demand_df["month"]   = pd.to_datetime(demand_df["month"], format="%Y-%m")
demand_df["jobs"]    = demand_df["jobs"].str.strip()
demand_df["country"] = demand_df["country"].str.strip()

DEMAND_JOB_TITLES = sorted(demand_df["jobs"].unique().tolist())
DEMAND_COUNTRIES  = sorted(demand_df["country"].unique().tolist())


# ── Prophet models ────────────────────────────────────────────────────────────

MODELS_PATH = "models/prophet_models.pkl"
prophet_models: dict = {}

def load_prophet_models():
    global prophet_models
    if not os.path.exists(MODELS_PATH):
        app.logger.warning(f"Prophet models not found at {MODELS_PATH}")
        return
    try:
        with open(MODELS_PATH, "rb") as f:
            prophet_models = pickle.load(f)
        app.logger.info(f"Loaded {len(prophet_models)} Prophet models.")
    except Exception as e:
        app.logger.error(f"Failed to load Prophet models: {e}")

load_prophet_models()

def _get_model(job_title: str, country: str):
    key = (job_title.strip(), country.strip())
    if key in prophet_models:
        return prophet_models[key]
    for (k_job, k_country), model in prophet_models.items():
        if k_job.strip() == key[0] and k_country.strip() == key[1]:
            return model
    return None

def _run_forecast(job_title: str, country: str, periods: int):
    model = _get_model(job_title, country)
    if model is None:
        raise ValueError(f"No trained model for job='{job_title}', country='{country}'.")

    hist = (
        demand_df[
            (demand_df["jobs"]    == job_title.strip()) &
            (demand_df["country"] == country.strip())
        ]
        .rename(columns={"month": "ds", "demand": "y"})
        .sort_values("ds")[["ds", "y"]]
        .reset_index(drop=True)
    )

    future   = model.make_future_dataframe(periods=periods, freq="MS")
    forecast = model.predict(future)

    last_hist_date = hist["ds"].max() if not hist.empty else pd.Timestamp("1900-01-01")
    fc_future = (
        forecast[forecast["ds"] > last_hist_date]
        [["ds", "yhat", "yhat_lower", "yhat_upper"]]
        .head(periods)
        .reset_index(drop=True)
    )
    return hist, fc_future


# ── Metrics ───────────────────────────────────────────────────────────────────

def dashboard_metrics():
    avg_salary = df["salary"].mean()
    return {
        "total_jobs":      len(df),
        "avg_salary":      int(round(avg_salary, 0)) if not pd.isna(avg_salary) else 0,
        "total_countries": int(df["country"].nunique()),
        "top_role":        df["job_title"].mode()[0],
        "remote_pct":      round((df["remote_work_option"].str.strip().str.lower() == "yes").mean() * 100, 1),
        "top_country":     df["country"].value_counts().idxmax(),
        "top_sector":      df["sector"].value_counts().idxmax(),
        "top_industry":    df["industry"].value_counts().idxmax(),
    }


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
@login_required
def dashboard():
    m = dashboard_metrics()
    return render_template("dashboard.html", **m,
        user_name=current_user.name, user_email=current_user.email)

@app.route("/forecast")
@login_required
def forecast():
    return render_template("forecast.html",
        job_titles=DEMAND_JOB_TITLES, countries=DEMAND_COUNTRIES,
        user_name=current_user.name,  user_email=current_user.email)


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.route("/auth/register", methods=["POST"])
def register():
    data     = request.get_json()
    name     = (data.get("name")     or "").strip()
    email    = (data.get("email")    or "").strip().lower()
    password = (data.get("password") or "").strip()

    if not name or not email or not password:
        return jsonify({"ok": False, "error": "All fields are required."}), 400
    if len(password) < 8:
        return jsonify({"ok": False, "error": "Password must be at least 8 characters."}), 400
    if "@" not in email:
        return jsonify({"ok": False, "error": "Please enter a valid email address."}), 400

    with get_db() as conn:
        if conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone():
            return jsonify({"ok": False, "error": "An account with this email already exists."}), 409
        hashed = generate_password_hash(password)
        cursor = conn.execute(
            "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
            (name, email, hashed)
        )
        conn.commit()
        user_id = cursor.lastrowid

    login_user(User(user_id, name, email), remember=True)
    return jsonify({"ok": True, "redirect": url_for("dashboard")}), 201

@app.route("/auth/login", methods=["POST"])
def login():
    data     = request.get_json()
    email    = (data.get("email")    or "").strip().lower()
    password = (data.get("password") or "").strip()

    if not email or not password:
        return jsonify({"ok": False, "error": "Email and password are required."}), 400

    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

    if not row or not check_password_hash(row["password"], password):
        return jsonify({"ok": False, "error": "Incorrect email or password."}), 401

    login_user(User(row["id"], row["name"], row["email"]), remember=True)
    return jsonify({"ok": True, "redirect": url_for("dashboard")}), 200

@app.route("/auth/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))


# ── Public API ────────────────────────────────────────────────────────────────

@app.route("/api/public/hero-stats")
def public_hero_stats():
    return jsonify(dashboard_metrics())

@app.route("/api/public/top-roles")
def public_top_roles():
    roles = df["job_title"].value_counts().head(5)
    return jsonify({"labels": roles.index.tolist(), "values": roles.values.tolist()})

@app.route("/api/public/jobs-by-country")
def public_jobs_by_country():
    counts = df["country"].value_counts().sort_values(ascending=False)
    return jsonify({"labels": counts.index.tolist(), "values": counts.values.tolist()})


# ── Dashboard API ─────────────────────────────────────────────────────────────

ORIGINAL_COLS = [
    "job_id", "job_title", "company", "company_type", "location",
    "country", "industry", "salary_range_usd", "experience_level",
    "job_type", "posting_date", "application_deadline",
    "required_education", "languages_required", "remote_work_option",
    "benefits", "contact_email", "application_method", "sector",
]

@app.route("/api/jobs")
@login_required
def jobs_data():
    cols = [c for c in ORIGINAL_COLS if c in df.columns]
    return jsonify(df[cols].fillna("—").to_dict(orient="records"))

@app.route("/api/chart/jobs-by-country")
@login_required
def jobs_by_country():
    counts = df["country"].value_counts().sort_values(ascending=False)
    return jsonify({"labels": counts.index.tolist(), "values": counts.values.tolist()})

@app.route("/api/chart/salary-trend")
@login_required
def salary_trend():
    trend = df.dropna(subset=["year", "salary"]).groupby("year")["salary"].mean().sort_index()
    return jsonify({
        "labels": trend.index.astype(int).astype(str).tolist(),
        "values": [round(v, 2) for v in trend.values.tolist()],
    })

@app.route("/api/chart/jobs-by-role")
@login_required
def jobs_by_role():
    roles = df["job_title"].value_counts().head(10)
    return jsonify({"labels": roles.index.tolist(), "values": roles.values.tolist()})

@app.route("/api/chart/jobs-by-experience")
@login_required
def jobs_by_experience():
    exp = df["experience_level"].value_counts()
    return jsonify({"labels": exp.index.tolist(), "values": exp.values.tolist()})

@app.route("/api/chart/jobs-by-type")
@login_required
def jobs_by_type():
    jtype = df["job_type"].value_counts()
    return jsonify({"labels": jtype.index.tolist(), "values": jtype.values.tolist()})

@app.route("/api/chart/jobs-by-sector")
@login_required
def jobs_by_sector():
    sector = df["sector"].value_counts()
    return jsonify({"labels": sector.index.tolist(), "values": sector.values.tolist()})

@app.route("/api/chart/jobs-by-industry")
@login_required
def jobs_by_industry():
    industry = df["industry"].value_counts().head(12)
    return jsonify({"labels": industry.index.tolist(), "values": industry.values.tolist()})

@app.route("/api/chart/remote-split")
@login_required
def remote_split():
    remote = df["remote_work_option"].value_counts()
    return jsonify({"labels": remote.index.tolist(), "values": remote.values.tolist()})

@app.route("/api/chart/salary-by-experience")
@login_required
def salary_by_experience():
    sal = df.dropna(subset=["salary"]).groupby("experience_level")["salary"].mean().round(2)
    return jsonify({"labels": sal.index.tolist(), "values": sal.values.tolist()})


# ── Forecast API ──────────────────────────────────────────────────────────────

@app.route("/api/forecast/meta")
@login_required
def forecast_meta():
    model_keys = [{"job_title": k[0], "country": k[1]} for k in prophet_models.keys()]
    return jsonify({
        "job_titles": DEMAND_JOB_TITLES,
        "countries":  DEMAND_COUNTRIES,
        "model_keys": model_keys,
    })

@app.route("/api/forecast/countries-for-job/<path:job_title>")
@login_required
def countries_for_job(job_title):
    jt = job_title.strip()
    countries_with_data  = set(demand_df[demand_df["jobs"] == jt]["country"].unique())
    countries_with_model = {k[1] for k in prophet_models.keys() if k[0] == jt}
    return jsonify({"job_title": jt, "countries": sorted(countries_with_data & countries_with_model)})

@app.route("/api/forecast/historical")
@login_required
def forecast_historical():
    job_title = (request.args.get("job_title") or "").strip()
    country   = (request.args.get("country")   or "").strip()

    if not job_title or not country:
        return jsonify({"error": "job_title and country are required."}), 400

    subset = demand_df[
        (demand_df["jobs"] == job_title) & (demand_df["country"] == country)
    ].sort_values("month")

    if subset.empty:
        return jsonify({"error": f"No historical data for ('{job_title}', '{country}')."}), 404

    return jsonify({
        "job_title": job_title,
        "country":   country,
        "labels":    subset["month"].dt.strftime("%Y-%m").tolist(),
        "values":    subset["demand"].tolist(),
    })

@app.route("/api/forecast/predict")
@login_required
def forecast_predict():
    job_title = (request.args.get("job_title") or "").strip()
    country   = (request.args.get("country")   or "").strip()
    try:
        periods = max(1, min(int(request.args.get("periods", 12)), 36))
    except ValueError:
        periods = 12

    if not job_title or not country:
        return jsonify({"error": "job_title and country are required."}), 400

    try:
        hist, fc = _run_forecast(job_title, country, periods)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Forecast failed: {str(e)}"}), 500

    return jsonify({
        "job_title":  job_title,
        "country":    country,
        "periods":    periods,
        "historical": {
            "labels": hist["ds"].dt.strftime("%Y-%m").tolist(),
            "values": hist["y"].tolist(),
        },
        "forecast": {
            "labels": fc["ds"].dt.strftime("%Y-%m").tolist(),
            "values": fc["yhat"].round(2).tolist(),
            "lower":  fc["yhat_lower"].round(2).tolist(),
            "upper":  fc["yhat_upper"].round(2).tolist(),
        },
    })

@app.route("/api/forecast/summary")
@login_required
def forecast_summary():
    results = []
    for (job_title, country), model in prophet_models.items():
        hist = demand_df[
            (demand_df["jobs"] == job_title) & (demand_df["country"] == country)
        ].sort_values("month")

        last_demand = int(hist["demand"].iloc[-1]) if not hist.empty else None
        last_month  = hist["month"].iloc[-1].strftime("%Y-%m") if not hist.empty else None

        try:
            future   = model.make_future_dataframe(periods=12, freq="MS")
            forecast = model.predict(future)
            last_ts  = hist["month"].max() if not hist.empty else pd.Timestamp("1900-01-01")
            fc_slice = forecast[forecast["ds"] > last_ts].head(12)

            next_val = round(float(fc_slice["yhat"].iloc[0]),  1) if not fc_slice.empty else None
            end_val  = round(float(fc_slice["yhat"].iloc[-1]), 1) if not fc_slice.empty else next_val
            diff     = (end_val - next_val) if (next_val is not None and end_val is not None) else 0
            trend    = "up" if diff > 0.5 else ("down" if diff < -0.5 else "stable")
        except Exception:
            next_val, trend = None, "stable"

        results.append({
            "job_title":   job_title,
            "country":     country,
            "last_demand": last_demand,
            "last_month":  last_month,
            "next_month":  next_val,
            "trend_12m":   trend,
        })

    results.sort(key=lambda x: (x["job_title"], x["country"]))
    return jsonify(results)

@app.route("/api/forecast/user-predict", methods=["POST"])
@login_required
def user_predict():
    data      = request.get_json(silent=True) or {}
    job_title = (data.get("job_title") or "").strip()
    country   = (data.get("country")   or "").strip()
    try:
        periods = max(1, min(int(data.get("periods", 6)), 36))
    except (ValueError, TypeError):
        periods = 6

    if not job_title:
        return jsonify({"ok": False, "error": "Please select a job title."}), 400
    if not country:
        return jsonify({"ok": False, "error": "Please select a country."}), 400

    try:
        hist, fc = _run_forecast(job_title, country, periods)
    except ValueError as e:
        available = sorted({k[1] for k in prophet_models if k[0] == job_title})
        return jsonify({"ok": False, "error": str(e), "available_countries": available}), 404
    except Exception as e:
        return jsonify({"ok": False, "error": f"Prediction failed: {str(e)}"}), 500

    if fc.empty:
        return jsonify({"ok": False, "error": "Forecast produced no future data points."}), 500

    predictions = [
        {
            "month":  row["ds"].strftime("%Y-%m"),
            "demand": round(float(row["yhat"]),       1),
            "lower":  round(float(row["yhat_lower"]), 1),
            "upper":  round(float(row["yhat_upper"]), 1),
        }
        for _, row in fc.iterrows()
    ]

    demands  = [p["demand"] for p in predictions]
    peak_idx = demands.index(max(demands))
    low_idx  = demands.index(min(demands))
    diff     = demands[-1] - demands[0]
    trend    = "up" if diff > 0.5 else ("down" if diff < -0.5 else "stable")

    return jsonify({
        "ok":          True,
        "job_title":   job_title,
        "country":     country,
        "periods":     periods,
        "predictions": predictions,
        "summary": {
            "avg_demand":  round(sum(demands) / len(demands), 1),
            "peak_month":  predictions[peak_idx]["month"],
            "peak_demand": predictions[peak_idx]["demand"],
            "low_month":   predictions[low_idx]["month"],
            "low_demand":  predictions[low_idx]["demand"],
            "trend":       trend,
        },
    })


if __name__ == "__main__":
    app.run(debug=True)