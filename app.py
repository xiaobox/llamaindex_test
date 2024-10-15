from flask import Flask, render_template, jsonify, request
from main import TestLlamaIndex
import unittest
import io
import sys

app = Flask(__name__)

@app.route('/')
def index():
    test_cases = [method for method in dir(TestLlamaIndex) if method.startswith('test_')]
    return render_template('index.html', test_cases=test_cases)

@app.route('/run_test', methods=['POST'])
def run_test():
    test_name = request.json['test_name']
    
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    suite = unittest.TestSuite()
    suite.addTest(TestLlamaIndex(test_name))
    runner = unittest.TextTestRunner(stream=captured_output)
    result = runner.run(suite)
    
    sys.stdout = sys.__stdout__
    
    output = captured_output.getvalue()
    
    return jsonify({
        'success': result.wasSuccessful(),
        'output': output
    })

if __name__ == '__main__':
    app.run(debug=True, port=8000, host='0.0.0.0')
