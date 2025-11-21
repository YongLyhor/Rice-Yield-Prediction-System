import http.server
import socketserver
import json
import joblib
import numpy as np
import webbrowser
from threading import Timer
from urllib.parse import parse_qs
import os

PORT = 8000

# Load models and scaler
scaler = joblib.load('standard_scaler.joblib')
sklearn_model = joblib.load('scikit_learn_yield_linear.joblib')
scratch_model = joblib.load('from_scratch_yield_model.joblib')

models = {
    'sklearn': sklearn_model,
    'scratch': scratch_model
}

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/UI.html':
            self.path = '/UI.html'
            return http.server.SimpleHTTPRequestHandler.do_GET(self)
        else:
            return http.server.SimpleHTTPRequestHandler.do_GET(self)
    
    def do_POST(self):
        if self.path == '/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            try:
                # Get input values
                n_req = float(data['n_req'])
                p_req = float(data['p_req'])
                k_req = float(data['k_req'])
                model_type = data['model_type']
                
                # Create input array
                input_data = np.array([[n_req, p_req, k_req]])
                
                # Scale the input using the standard scaler
                input_scaled = scaler.transform(input_data)
                
                # Select model and predict
                selected_model = models[model_type]
                prediction = selected_model.predict(input_scaled)[0]
                
                response = {
                    'success': True,
                    'prediction': round(float(prediction), 2),
                    'model_used': 'Scikit-Learn Linear Regression' if model_type == 'sklearn' else 'SGD Regressor (From Scratch)'
                }
            except Exception as e:
                response = {
                    'success': False,
                    'error': str(e)
                }
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

def open_browser():
    webbrowser.open_new(f'http://localhost:{PORT}/UI.html')

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
        print(f"Server running at http://localhost:{PORT}/")
        print("Opening browser...")
        Timer(1, open_browser).start()
        httpd.serve_forever()
