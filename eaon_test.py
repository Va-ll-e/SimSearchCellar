# Aeon requires python v.12
# pip install aeon[all]
from aeon.datasets import load_airline

y = load_airline()

print(len(y))

print(y[:5])