from flask import Flask, render_template, request
import pandas as pd
import os

app = Flask(__name__,
            template_folder=os.path.join("..", "templates"),
            static_folder=os.path.join("..", "static"))

# Import your functions
from f1_data_pipeline import get_meetings, get_drivers, get_sessions, get_pit, get_weather

@app.route("/", methods=["GET", "POST"])
def index():
    data = None
    if request.method == "POST":
        # Get the type of data to fetch based on user input
        data_type = request.form.get("data_type")
        
        # Fetch data based on selection
        if data_type == "meetings":
            data = get_meetings()
        elif data_type == "drivers":
            data = get_drivers()
        elif data_type == "sessions":
            data = get_sessions()
        elif data_type == "pit":
            data = get_pit()
        elif data_type == "weather":
            data = get_weather()
        
        # Convert DataFrame to HTML for rendering
        if data is not None:
            data = data.to_html(classes="data-table", index=False)

    return render_template("index.html", data=data)

if __name__ == "__main__":
    app.run(debug=True)
