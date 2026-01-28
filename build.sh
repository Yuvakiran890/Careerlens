#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

# Pre-download NLTK data during build so it's ready when the app starts
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
