from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        # ── String variable ──────────────────────────
        name       = str(request.form.get("name", ""))
        # ── Integer variable ─────────────────────────
        age        = int(request.form.get("age", 0))
        # ── Float variable ───────────────────────────
        height     = float(request.form.get("height", 0.0))
        # ── Boolean variable ─────────────────────────
        is_student = request.form.get("is_student") == "on"
        # ── List variable ────────────────────────────
        hobbies    = request.form.get("hobbies", "").split(",")
        hobbies    = [h.strip() for h in hobbies if h.strip()]
        # ── Dictionary variable ──────────────────────
        profile    = {
            "name":       name,
            "age":        age,
            "height_cm":  height,
            "is_student": is_student,
            "hobbies":    hobbies,
            "hobby_count": len(hobbies),
        }

        result = profile

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
