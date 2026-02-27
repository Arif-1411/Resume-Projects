from flask import Flask, render_template, request

# ── importing custom modules ──────────────────────────────────────
import math_utils
import string_utils
import date_utils

# ── importing Python standard library modules ─────────────────────
import math
import random
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    results = {}

    if request.method == "POST":
        a        = float(request.form.get("a", 10))
        b        = float(request.form.get("b", 3))
        text     = request.form.get("text", "racecar")
        year     = int(request.form.get("year", 2000))
        dob_year = int(request.form.get("dob_year", 1995))

        # ── math_utils (custom module) ────────────────────────────
        results["math"] = {
            "add":      math_utils.add(a, b),
            "subtract": math_utils.subtract(a, b),
            "multiply": math_utils.multiply(a, b),
            "divide":   math_utils.divide(a, b),
            "power":    math_utils.power(a, b),
            "is_prime": math_utils.is_prime(int(a)),
        }

        # ── string_utils (custom module) ──────────────────────────
        results["string"] = {
            "original":    text,
            "reversed":    string_utils.reverse_string(text),
            "word_count":  string_utils.count_words(text),
            "palindrome":  string_utils.is_palindrome(text),
            "title_case":  string_utils.title_case(text),
            "vowels":      string_utils.count_vowels(text),
        }

        # ── date_utils (custom module) ────────────────────────────
        results["date"] = {
            "today":          date_utils.today(),
            "time":           date_utils.current_time(),
            "age":            date_utils.age_from_year(dob_year),
            "days_new_year":  date_utils.days_until_new_year(),
            "day_of_week":    date_utils.day_of_week(year, 1, 1),
        }

        # ── Python built-in modules ───────────────────────────────
        results["stdlib"] = {
            "math_sqrt":   round(math.sqrt(a), 4),
            "math_pi":     round(math.pi, 6),
            "math_floor":  math.floor(a),
            "math_ceil":   math.ceil(a),
            "random_int":  random.randint(1, 100),
            "random_choice": random.choice(["Python", "Flask", "Django", "FastAPI"]),
            "os_cwd":      os.getcwd(),
            "os_platform": os.name,
        }

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True, port=5006)
