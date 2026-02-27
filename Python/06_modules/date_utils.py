# date_utils.py  — custom module using Python's built-in datetime module

from datetime import datetime, date

def today():
    return date.today().strftime("%d %B %Y")

def current_time():
    return datetime.now().strftime("%H:%M:%S")

def age_from_year(birth_year):
    return date.today().year - birth_year

def days_until_new_year():
    today = date.today()
    ny    = date(today.year + 1, 1, 1)
    return (ny - today).days

def day_of_week(year, month, day):
    return date(year, month, day).strftime("%A")
