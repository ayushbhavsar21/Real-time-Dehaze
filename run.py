from app import create_app

app = create_app()

# gunicorn -w 4 -b 0.0.0.0:8000 app:app
