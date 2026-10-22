# -----------------------------------------------------------------------------------------------------------------------
# IMPORTS & MODULES
# -----------------------------------------------------------------------------------------------------------------------

from pathlib import Path

from flask import jsonify

from titanic_project.flask_app import create_app

# -----------------------------------------------------------------------------------------------------------------------
# APPLICATION: Main entrypoint
# -----------------------------------------------------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_PATH = PACKAGE_DIR / "data" / "titanic_800.csv"

app = create_app(DATA_PATH)

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=8001,
        debug=True,
        use_reloader=False,
    )
