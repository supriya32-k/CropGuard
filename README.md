# CropGuard: Smart Fertilizer & Disease Advisor

CropGuard is a web application that helps farmers and agriculturalists make informed decisions about fertilizer use and plant disease management using AI and data-driven recommendations.

## Features
- **Fertilizer Recommendation:**
  - Suggests optimal fertilizer adjustments based on soil N, P, K values and selected crop.
- **Plant Disease Detection:**
  - Upload a leaf image to detect plant diseases using a deep learning model.
  - Provides disease information, causes, and prevention tips.

## Requirements
- Python 3.8+
- pip
- See `requirements.txt` for all dependencies

## Installation
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd Harvestify/app
   ```
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your API keys in `config.py` (see sample in the file).

## Usage
1. Run the Flask app:
   ```bash
   python app.py
   ```
2. Open your browser and go to `http://localhost:5000`
3. Use the Fertilizer or Disease features from the navigation bar.

## Project Structure
```
Harvestify/app/
├── app.py                # Main Flask application
├── config.py             # API keys and configuration
├── utils/
│   ├── disease.py        # Disease info dictionary
│   ├── fertilizer.py     # Fertilizer info dictionary
│   └── model.py          # PyTorch model architecture
├── models/
│   └── plant_disease_model.pth  # Trained disease detection model
├── Data/
│   └── fertilizer.csv    # Fertilizer data
├── templates/            # HTML templates
├── static/
│   ├── css/              # CSS files
│   ├── images/           # Images for UI
│   └── temp_uploads/     # Temporary image uploads
├── requirements.txt      # Python dependencies
├── Procfile              # For deployment (e.g., Heroku)
└── Runtime.txt           # Python version for deployment
```

## Credits
- Built with [Flask](https://flask.palletsprojects.com/), [PyTorch](https://pytorch.org/), and [scikit-learn](https://scikit-learn.org/)
- UI inspired by modern glassmorphism design

## License
This project is for educational purposes.
