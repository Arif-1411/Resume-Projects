# string_utils.py  — custom module for string operations

def reverse_string(s):
    return s[::-1]

def count_words(s):
    return len(s.split())

def is_palindrome(s):
    clean = s.lower().replace(" ", "")
    return clean == clean[::-1]

def title_case(s):
    return s.title()

def count_vowels(s):
    vowels = "aeiouAEIOU"
    return sum(1 for c in s if c in vowels)
