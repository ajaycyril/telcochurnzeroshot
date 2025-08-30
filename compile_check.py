import py_compile, sys
try:
    py_compile.compile('app.py', doraise=True)
    print('COMPILE_OK')
except Exception as e:
    print('COMPILE_FAIL', e)
    sys.exit(2)
