from flask import Flask, render_template, request

app = Flask(__name__)

# ── Base Class ────────────────────────────────────────────────────
class Animal:
    def __init__(self, name, species):
        self.name    = name        # instance variable
        self.species = species
        self.tricks  = []          # mutable instance variable

    def speak(self):
        return f"{self.name} makes a sound."

    def learn_trick(self, trick):
        self.tricks.append(trick)

    def info(self):
        return {
            "name":    self.name,
            "species": self.species,
            "tricks":  self.tricks,
        }

# ── Subclass — Inheritance ────────────────────────────────────────
class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, species="Dog")  # calling parent __init__
        self.breed = breed

    def speak(self):                           # method overriding
        return f"{self.name} says: Woof! 🐶"

    def fetch(self, item):
        return f"{self.name} fetched the {item}!"

# ── Another Subclass ──────────────────────────────────────────────
class Cat(Animal):
    def __init__(self, name, indoor):
        super().__init__(name, species="Cat")
        self.indoor = indoor

    def speak(self):                           # method overriding
        return f"{self.name} says: Meow! 🐱"

    def info(self):                            # extending parent method
        data = super().info()
        data["indoor"] = self.indoor
        return data

# ── Class with class variable ─────────────────────────────────────
class BankAccount:
    bank_name   = "PyBank"                    # class variable (shared)
    total_accounts = 0

    def __init__(self, owner, balance=0):
        self.owner   = owner
        self.balance = float(balance)
        self.history = []
        BankAccount.total_accounts += 1

    def deposit(self, amount):
        self.balance += amount
        self.history.append(f"+ ₹{amount}")
        return self.balance

    def withdraw(self, amount):
        if amount > self.balance:
            return None             # insufficient funds
        self.balance -= amount
        self.history.append(f"- ₹{amount}")
        return self.balance

    def statement(self):
        return {
            "owner":   self.owner,
            "bank":    self.bank_name,
            "balance": round(self.balance, 2),
            "history": self.history,
        }


@app.route("/", methods=["GET", "POST"])
def index():
    results = {}

    if request.method == "POST":
        # Dog
        dog_name  = request.form.get("dog_name", "Buddy")
        dog_breed = request.form.get("dog_breed", "Labrador")
        dog_trick = request.form.get("dog_trick", "sit")

        dog = Dog(dog_name, dog_breed)
        dog.learn_trick(dog_trick)
        dog.learn_trick("shake hands")
        results["dog"] = {
            "speak":   dog.speak(),
            "fetch":   dog.fetch("ball"),
            "info":    dog.info(),
            "breed":   dog.breed,
            "is_dog":  isinstance(dog, Animal),   # isinstance check
        }

        # Cat
        cat_name   = request.form.get("cat_name", "Whiskers")
        cat_indoor = request.form.get("cat_indoor") == "on"

        cat = Cat(cat_name, cat_indoor)
        results["cat"] = {
            "speak": cat.speak(),
            "info":  cat.info(),
        }

        # BankAccount
        owner   = request.form.get("owner", "Alice")
        balance = float(request.form.get("balance", 1000))
        dep     = float(request.form.get("deposit", 500))
        wdraw   = float(request.form.get("withdraw", 200))

        acc = BankAccount(owner, balance)
        acc.deposit(dep)
        acc.withdraw(wdraw)
        results["bank"] = acc.statement()
        results["total_accounts"] = BankAccount.total_accounts

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True, port=5005)
