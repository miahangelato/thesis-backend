#!/bin/bash
# entrypoint.sh - Debugging script for Railway deployment

echo "=== Environment Info ==="
echo "Platform: $(uname -a)"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "Directory listing:"
ls -la

echo "=== Django Info ==="
echo "Running migrations..."
python manage.py migrate

echo "=== Starting Gunicorn ==="
gunicorn backend.wsgi:application --bind 0.0.0.0:$PORT
