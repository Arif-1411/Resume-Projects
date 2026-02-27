from flask import Flask, render_template, request

app = Flask(__name__)

# ── 1. Basic function ─────────────────────────────────────────────
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# ── 2. Function with return value ─────────────────────────────────
def calculate_bmi(weight_kg, height_cm):
    height_m = height_cm / 100
    bmi = round(weight_kg / (height_m ** 2), 1)
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    return {"bmi": bmi, "category": category}

# ── 3. Function returning multiple values ──────────────────────────
def min_max_avg(numbers):
    total = 0
    for n in numbers:
        total += n
    return min(numbers), max(numbers), round(total / len(numbers), 2)

# ── 4. Function with *args ─────────────────────────────────────────
def total_price(*prices):
    return round(sum(prices), 2)

# ── 5. Function with **kwargs ──────────────────────────────────────
def build_profile(**kwargs):
    return kwargs

# ── 6. Recursive function ──────────────────────────────────────────
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# ── 7. Lambda function ─────────────────────────────────────────────
celsius_to_fahrenheit = lambda c: round(c * 9/5 + 32, 1)


@app.route("/", methods=["GET", "POST"])
def index():
    results = {}

    if request.method == "POST":
        name       = request.form.get("name", "World")
        greeting   = request.form.get("greeting", "Hello")
        weight     = float(request.form.get("weight", 70))
        height     = float(request.form.get("height", 170))
        nums_raw   = request.form.get("numbers", "10,20,30,40,50")
        prices_raw = request.form.get("prices", "99.9,149.5,59.0")
        n_fact     = int(request.form.get("factorial_n", 6))
        temp_c     = float(request.form.get("temp_c", 37))

        numbers = [float(x.strip()) for x in nums_raw.split(",") if x.strip()]
        prices  = [float(x.strip()) for x in prices_raw.split(",") if x.strip()]
        mn, mx, avg = min_max_avg(numbers)

        results = {
            "greet":      greet(name, greeting),
            "bmi":        calculate_bmi(weight, height),
            "min":        mn, "max": mx, "avg": avg,
            "total":      total_price(*prices),
            "profile":    build_profile(name=name, weight=weight, height=height),
            "factorial":  {"n": n_fact, "result": factorial(n_fact)},
            "temp":       {"c": temp_c, "f": celsius_to_fahrenheit(temp_c)},
        }

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True, port=5004)
