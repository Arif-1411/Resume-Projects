from flask import Flask, render_template, request

app = Flask(__name__)

def check_eligibility(age, income, has_degree, experience):
    results = []

    # ── if / elif / else ─────────────────────────────────────────
    # Age check
    if age < 18:
        age_status = "Too young (under 18)"
        age_ok = False
    elif age > 60:
        age_status = "Above retirement age"
        age_ok = False
    else:
        age_status = "Age eligible"
        age_ok = True
    results.append({"check": "Age", "status": age_status, "ok": age_ok})

    # Income check
    if income < 10000:
        inc_status = "Income too low"
        inc_ok = False
    elif income < 30000:
        inc_status = "Low income — partial benefits"
        inc_ok = True
    elif income < 100000:
        inc_status = "Middle income — full benefits"
        inc_ok = True
    else:
        inc_status = "High income — no subsidy"
        inc_ok = False
    results.append({"check": "Income", "status": inc_status, "ok": inc_ok})

    # ── and / or conditions ──────────────────────────────────────
    if has_degree and experience >= 2:
        job_status = "Qualified — degree + experience"
        job_ok = True
    elif has_degree and experience < 2:
        job_status = "Partially qualified — needs more experience"
        job_ok = True
    elif not has_degree and experience >= 5:
        job_status = "Qualified via experience (no degree needed)"
        job_ok = True
    else:
        job_status = "Not qualified"
        job_ok = False
    results.append({"check": "Job Eligibility", "status": job_status, "ok": job_ok})

    # ── Overall decision ─────────────────────────────────────────
    approved = age_ok and inc_ok and job_ok

    return results, approved


@app.route("/", methods=["GET", "POST"])
def index():
    results   = None
    approved  = None
    form_data = {}

    if request.method == "POST":
        age        = int(request.form.get("age", 0))
        income     = int(request.form.get("income", 0))
        has_degree = request.form.get("has_degree") == "on"
        experience = int(request.form.get("experience", 0))

        form_data = {"age": age, "income": income,
                     "has_degree": has_degree, "experience": experience}
        results, approved = check_eligibility(age, income, has_degree, experience)

    return render_template("index.html", results=results,
                           approved=approved, form_data=form_data)

if __name__ == "__main__":
    app.run(debug=True, port=5002)
