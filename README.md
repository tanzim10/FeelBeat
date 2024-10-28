# FeelBeat

A Music Recommendation System Based on Mood Detection

## Overview

FeelBeat is a project that introduces an innovative music recommendation system driven by facial emotion recognition technology. The system offers personalized, real-time music suggestions aligned with the user's current emotional state. While existing platforms lack this feature, our goal is to bridge this gap and provide a dedicated device application for users. We aim to enhance accuracy, improve real-time emotion detection, and empower users to select music that resonates with their specific emotional states. 

## Features

- **Mood Detection:** Detects user mood in real-time using facial emotion recognition.
- **Contextual Analysis:** Integrates additional contextual factors, such as weather and activity, for better personalization.
- **Hybrid Recommendation System:** Combines content-based filtering with mood-based contextual data to curate playlists.
- **User-Friendly Interface:** Simple and intuitive UI to enhance user engagement.

## How It Works

1. **Facial Emotion Recognition:** Captures facial expressions and uses pre-trained models (e.g., ResNet50, VGG19) to predict the user’s emotional state.
2. **Contextual Data Gathering:** Considers real-time contextual data such as weather, activity, and age.
3. **Hybrid Recommendation Engine:** Determines the user’s final mood using both facial recognition and contextual cues, applying content-based filtering for optimal music recommendations.
4. **Personalized Playlist Generation:** Based on the detected mood, a playlist of 10 songs that best matches the user's current emotional state is generated.

## Screenshots

### Homepage
![Homepage](https://drive.google.com/uc?export=view&id=1sokOKTJNMORY6tjr4BPgC25r49bB090D)

### Mood Detection in Action
![Mood Detection](https://drive.google.com/uc?export=view&id=1b_DxWwLZi6T3IMMa_9aj05ONSO9-BHTC)

### Suggested Playlist Based on Mood
![Mood-Based Playlist](https://drive.google.com/uc?export=view&id=1EP6p5Cq8H_I65HoTlOH2hkCBkY7Hf6me)

### Custom Resnet Architecture 
![Custom Resnet](https://drive.google.com/uc?export=view&id=1co0cvtBEdThOqwZnYFqjdnQe_ccV79w5)

> *Add your screenshots in the sections above to showcase the app's functionalities.*

## Datasets Used

- **FER2013** for emotion recognition.
- **Spotify Million Playlist Dataset** for song recommendations based on content features.
- **JAFFE, CK+, AffectNet** for supplementary emotion detection training data.
- **Custom Survey Dataset** of contextual data, including user activities, weather, and music preferences.

## Models and Techniques

- **Emotion Recognition Models**: Fine-tuned ResNet50, VGG19, Inception-ResNetV2, and EfficientNetV2 models for accurate facial emotion recognition.
- **Contextual Recommendation**: Random Forest model predicting mood based on contextual features like weather and user activity.
- **Hybrid Recommendation Engine**: Combines facial emotion recognition and context-aware predictions for high personalization.
- **Content-Based Filtering**: Utilizes cosine similarity to match the final mood with song features (e.g., tempo, genre) for playlist curation.

## Results

FeelBeat shows significant improvement in recommendation relevance, with up to 20% enhancement over traditional content-based systems.

| Model               | Accuracy (FER2013) | Accuracy (JAFFE) | Accuracy (CK+) |
|---------------------|--------------------|-------------------|----------------|
| ResNet50            | 72.5%              | 70%              | 70%            |
| VGG19               | 70.2%              | 69%              | 69%            |
| Inception-ResNetV2  | 61.7%              | -                | -              |

## Getting Started

1. **Clone the repository**
   ```bash
   https://github.com/tanzim10/FeelBeat.git
   ```
2.  **Install dependencies**
    ``` bash
    pip install -r requirements.txt
    ```
4.  **Run the Application**
    ```bash
    python3 manage.py runserver
    ```

## License
This project is licensed under the MIT License.

## Acknowledgments
 Special thanks to our supervisor, Mr. Intisar Tahmid Naheen, and the Department of Computer Science and Engineering, North South University.

 
   
