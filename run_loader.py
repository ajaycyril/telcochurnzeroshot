import os
import sys
import traceback
import app

print('Python executable:', sys.executable)
print('Current working dir:', os.getcwd())
print('app.py location:', os.path.abspath(app.__file__))
print('TELCO_CSV constant:', app.TELCO_CSV)
print('KAGGLE_USERNAME present:', 'KAGGLE_USERNAME' in os.environ)
print('KAGGLE_KEY present:', 'KAGGLE_KEY' in os.environ)

# check possible locations
paths = [os.path.abspath(app.TELCO_CSV), os.path.join(os.path.dirname(app.__file__), app.TELCO_CSV)]
for p in paths:
    print('Exists:', p, os.path.exists(p))

try:
    df = app.load_telco_data()
    print('Loaded DataFrame shape:', df.shape)
except Exception as e:
    print('Error while loading dataset:')
    traceback.print_exc()
    sys.exit(2)

print('Success')
