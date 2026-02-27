from flask import Flask, render_template, request

app = Flask(__name__)

def run_loops(numbers_raw, word):
    results = {}

    numbers = [int(x.strip()) for x in numbers_raw.split(",") if x.strip().lstrip("-").isdigit()]

    # ── for loop ─────────────────────────────────────
    squares = []
    for n in numbers:
        squares.append({"n": n, "square": n * n})
    results["for_squares"] = squares

    # ── while loop ───────────────────────────────────
    countdown = []
    i = 10
    while i >= 0:
        countdown.append(i)
        i -= 1
    results["while_countdown"] = countdown

    # ── for loop with range ───────────────────────────
    multiplication = []
    num = numbers[0] if numbers else 5
    for i in range(1, 11):
        multiplication.append({"i": i, "product": num * i})
    results["multiplication"] = multiplication
    results["mul_num"] = num

    # ── nested for loop ───────────────────────────────
    pairs = []
    small = numbers[:3] if len(numbers) >= 3 else [1, 2, 3]
    for a in small:
        for b in small:
            pairs.append({"a": a, "b": b, "sum": a + b})
    results["nested_pairs"] = pairs

    # ── for loop with enumerate ────────────────────────
    letters = []
    for index, char in enumerate(word.upper()):
        letters.append({"index": index, "char": char})
    results["enumerate"] = letters

    # ── list comprehension (compact loop) ─────────────
    evens = [n for n in numbers if n % 2 == 0]
    odds  = [n for n in numbers if n % 2 != 0]
    results["evens"] = evens
    results["odds"]  = odds

    return results


@app.route("/", methods=["GET", "POST"])
def index():
    results   = None
    numbers   = "3, 6, 9, 2, 7, 4, 8, 1"
    word      = "PYTHON"

    if request.method == "POST":
        numbers = request.form.get("numbers", numbers)
        word    = request.form.get("word", word)
        results = run_loops(numbers, word)

    return render_template("index.html", results=results, numbers=numbers, word=word)

if __name__ == "__main__":
    app.run(debug=True, port=5003)
