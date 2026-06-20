import http.server
import socketserver
import json
import urllib.parse
import os
import sys
import pandas as pd

# Ensure workspace root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.predict import InferencePipeline
from utils.pdf_generator import generate_incident_pdf_report

PORT = 8085
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class DashboardAPIHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
        
    def do_POST(self):
        if self.path == '/api/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                event = json.loads(post_data.decode('utf-8'))
                
                # Run prediction pipeline
                pipeline = InferencePipeline(models_dir="models")
                scored = pipeline.predict_one(event)
                
                # Generate PDF Operational report
                report_path = os.path.abspath(os.path.join("reports", "operational_incident_report.pdf"))
                generate_incident_pdf_report(scored, report_path)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(scored).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            
    def do_GET(self):
        if self.path == '/api/monitoring':
            self.serve_json_file(os.path.join("models", "monitoring_metrics.json"))
        elif self.path == '/api/history':
            self.serve_json_file(os.path.join("models", "prediction_logs.json"))
        elif self.path == '/api/experiment':
            self.serve_json_file(os.path.join("models", "experiment_history.json"))
        elif self.path == '/api/download_pdf':
            self.serve_pdf_file(os.path.join("reports", "operational_incident_report.pdf"))
        elif self.path == '/api/download_csv':
            self.generate_and_serve_csv(os.path.join("models", "prediction_logs.json"))
        elif self.path.startswith('/api/reports/'):
            filename = urllib.parse.unquote(os.path.basename(self.path))
            self.serve_image_file(os.path.join("reports", filename))
        else:
            # Fallback to serving static html
            super().do_GET()
            
    def serve_json_file(self, filepath):
        if os.path.exists(filepath):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            with open(filepath, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps([]).encode('utf-8'))
            
    def serve_pdf_file(self, filepath):
        if os.path.exists(filepath):
            self.send_response(200)
            self.send_header('Content-type', 'application/pdf')
            self.send_header('Content-Disposition', 'attachment; filename="operational_incident_report.pdf"')
            self.end_headers()
            with open(filepath, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()
            
    def serve_image_file(self, filepath):
        if os.path.exists(filepath):
            self.send_response(200)
            if filepath.endswith('.png'):
                self.send_header('Content-type', 'image/png')
            elif filepath.endswith('.jpg') or filepath.endswith('.jpeg'):
                self.send_header('Content-type', 'image/jpeg')
            self.end_headers()
            with open(filepath, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()
            
    def generate_and_serve_csv(self, logs_filepath):
        if os.path.exists(logs_filepath):
            try:
                with open(logs_filepath, 'r') as f:
                    logs = json.load(f)
                df = pd.DataFrame(logs)
                
                csv_data = df.to_csv(index=False)
                
                self.send_response(200)
                self.send_header('Content-type', 'text/csv')
                self.send_header('Content-Disposition', 'attachment; filename="prediction_history_report.csv"')
                self.end_headers()
                self.wfile.write(csv_data.encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

def run(port=PORT):
    # Ensure static HTML is placed correctly
    os.makedirs(DIRECTORY, exist_ok=True)
    
    Handler = DashboardAPIHandler
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"==============================================")
        print(f"Traffic Decision Platform Live at: http://localhost:{port}")
        print(f"==============================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()

if __name__ == "__main__":
    run()
