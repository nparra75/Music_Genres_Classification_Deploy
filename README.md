# Music_Genres_Classification_Deploy

This project classifies music genres using a Convolutional Neural Network (CNN) with MFCC features extracted from audio files.

## Project Structure
- `music_genre_model_with_mfcc.keras`: Trained model using MFCC features.
- `music_genre_model1.keras`, `music_genre_model2.keras`: Final models trained with Mel Spectograms.
- `scaler.pkl`, `scaler_new.pkl`: Pretrained scalers for data normalization.
- `Music_Genres_Classification_with_MFCC.ipynb`: Jupyter Notebook for experimentation.
- `Requirements.txt`: Python dependencies for the project.
- `app.py`: Streamlit web application for music genre prediction.

## Requirements
Install required Python libraries:
```bash
pip install -r Requirements.txt
```

## Usage

1. Clone this repository: git clone https://github.com/<your-username>/<repository-name>.git

2. Navigate to the project directory: cd <repository-name>

3. Run the Streamlit app: streamlit run app.py

## Dataset

The dataset used is the GTZAN dataset, which should be placed in the genres_original folder within the project directory.

## Results

The trained models achieve the following accuracies on the test set:
- "Model based on spectrograms 1": 0.8487,
- "Model based on spectrograms 2": 0.8498,
- "Model based on MFCC": 0.74
