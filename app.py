from flask import Flask, render_template, request, jsonify
from quickcommerce import main, chart  # ✅ Import from your file
app = Flask(__name__)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')
@app.route('/get-chart-data', methods=['POST'])
def get_chart_data():
    data = request.get_json()
    query = data.get('query', '')
    print(f"Received query: {query}")  # ✅ Debugging line to check the query
    values = chart(query)  # ✅ Call your function here
    return jsonify({"values": values})

if __name__ == '__main__':
    app.run(debug=True)

