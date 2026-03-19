import hmac, hashlib, json, os, traceback
import requests
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MS_APP_KEY  = 'bf786719-2152-4641-bf5f-7ad282e8d585'
MS_HMAC_KEY = '92b040fc-3c51-46ce-8870-202dfd888d7d'
MS_URL      = 'https://cloud.myscript.com/api/v4.0/iink/batch'

@app.route('/')
def index():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'canvas.html')
    return send_file(path)

@app.route('/recognise', methods=['POST'])
def recognise():
    try:
        data = request.get_json(force=True)
        body = {
            "xDPI": 96,
            "yDPI": 96,
            "contentType": "Math",
            "strokeGroups": data.get("strokeGroups", [])
        }
        body_str = json.dumps(body, separators=(',',':'))

        sig = hmac.new(
            (MS_APP_KEY + MS_HMAC_KEY).encode('utf-8'),
            msg=body_str.encode('utf-8'),
            digestmod=hashlib.sha512
        ).hexdigest()

        headers = {
            'Accept':         'application/x-latex',
            'Content-Type':   'application/json',
            'applicationKey': MS_APP_KEY,
            'hmac':           sig,
        }

        r = requests.post(MS_URL, headers=headers, data=body_str, timeout=15)
        print(f'[{r.status_code}] {r.text[:200]}')

        if not r.ok:
            return jsonify({'error': r.text}), r.status_code

        # Response is plain LaTeX text when Accept: application/x-latex
        return jsonify({'label': r.text.strip()})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print('\n  InkCluster at http://localhost:5000\n')
    app.run(host='0.0.0.0', port=5000, debug=False)