import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load
import requests
from django.contrib.auth.models import User
from User.models import Profile

# Load pre-trained models and data

global cascade, resnet50, data
cascade = cv2.CascadeClassifier("/Users/tanzimfarhan/Desktop/FeelBeat/sub_main/FER_SystemModule/haarcascade_frontalface_default.xml")
resnet50 = keras.models.load_model('/Users/tanzimfarhan/Desktop/FeelBeat/FB_Web/FeelBeat/fer/utilities/SOA-EPOCHS_150-DROPOUT_0.3-test_acc_0.659.h5', compile=False)
label_map = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
data = pd.read_csv('/Users/tanzimfarhan/Desktop/FeelBeat/sub_main/FER_SystemModule/spotify_1M_mood3.csv')
pk = requests.user.username
user = User.objects.get(username=pk)
profile = Profile.objects.get(user=user)



def process_image(image_np, activity):
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype("float32") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)  # Add batch dimension
        roi_gray = np.expand_dims(roi_gray, axis=-1)  # Add channel dimension

        prediction = resnet50.predict(roi_gray)
        label = np.argmax(prediction, axis=1)
        emotion = label_map[label[0]]
        confidence = np.max(prediction)  # passing the max probability as confidence
        
        emotion, recommended_songs = hybrid_helper(activity, emotion, confidence)  # Passing confidence instead of raw prediction

        print("Predicted Emotion:", emotion)
        print("Recommended Songs:")
        features = ['valence', 'tempo', 'danceability', 'energy','instrumentalness','liveness']
        songs_df = pd.read_csv('/Users/tanzimfarhan/Desktop/FeelBeat/sub_main/FER_SystemModule/spotify_1M_mood3.csv')

        return emotion, recommended_songs

    return None, None  # Return None if no faces are detected


def context_aware(data, model_path):
    try:
        rf_classifier = load(model_path)  # Ensure the model_path points directly to a .pkl file
    except Exception as e:
        print(f"Failed to load the model. Error: {e}")
        return None, None
    try:
        # Ensure the input data is in the correct format, e.g., DataFrame or NumPy array
        if not isinstance(data, (np.ndarray, pd.DataFrame)):
            print("Input data is not in a valid format.")
            return None, None

        # Predict the probabilities of the classes
        prediction_probabilities = rf_classifier.predict_proba(data)
        # Extract the class with the highest probability
        mood_index = np.argmax(prediction_probabilities, axis=1)[0]
        mood = rf_classifier.classes_[mood_index]
        confidence = prediction_probabilities[0][mood_index]
        print(f"Predicted mood for Contextual: {mood} with confidence: {confidence}")

        return mood, confidence
    except AttributeError:
        print("The loaded model does not support probability prediction.")
        return None, None
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None, None



def content_based_recommendations(mood, songs_df, features, num_recommendations=10, top_n=100):
    moods_map = {
        'angry': 'calm',
        'disgusted': 'energetic',
        'fear': 'calm',
        'happy': 'happy',
        'neutral': 'neutral',
        'sad': 'sad',
        'surprise': 'energetic'
    }
    if mood in moods_map:
        mood = moods_map[mood]
        mood_songs = songs_df[songs_df['mood'] == mood]

        # Calculate the mean feature values of the mood songs
        mood_profile = mood_songs[features].mean().values.reshape(1, -1)

        # Calculate the similarity between the mood profile and all songs
        similarities = cosine_similarity(songs_df[features], mood_profile).flatten()

        # Create a temporary DataFrame to hold songs and their similarity scores
        temp_df = songs_df.copy()
        temp_df['similarity'] = similarities

        # Select top N most similar songs
        top_similar_songs = temp_df.sort_values(by='similarity', ascending=False).head(top_n)

        # Randomly sample num_recommendations songs from the top N similar songs
        if len(top_similar_songs) >= num_recommendations:
            recommendations = top_similar_songs.sample(n=num_recommendations)
        else:
            recommendations = top_similar_songs
        
        print(f"Recommendations for :",recommendations,similarities)

        return recommendations[['track_name', 'artist_name']]
    
    else:
        print(f"No mood found for: {mood}")
        return pd.DataFrame()

    


def determine_mood(data, context_model_path, prediction_probabilities,mood):
    # Get predictions from both models
    context_mood, context_confidence = context_aware(data, context_model_path)
    fer_confidence = prediction_probabilities
    print(f"Face Emotion Recognition Confidence: {fer_confidence}")
    
    # Determine which mood to use based on confidence
    if context_confidence > fer_confidence:
        return context_mood
    else:
        fer_mood = mood
        return fer_mood

def hybrid_recommendation(mood,predictions,data, songs_df,num_recommendations=10, top_n=100):
    # Determine the mood based on the highest confidence
    context = pd.read_csv('/Users/tanzimfarhan/Desktop/FeelBeat/FB_Web/FeelBeat/fer/utilities/retrained_rf_model.pkl')
   
    final_mood = determine_mood(data, context,predictions,mood)
    
    # Get recommendations based on the mood
    features = ['valence', 'tempo', 'danceability', 'energy','instrumentalness','liveness']
    recommendations = content_based_recommendations(final_mood, songs_df, features, num_recommendations, top_n)
    
    
    return final_mood, recommendations

def hybrid_helper(activity,mood,predictions):
    api_key = '908ba200454f4d82926144343241406' 
    city = 'Dhaka'
    weather_data_1 = 'Clear'
    activity_map = {
            'Working/Studying': 0, 
            'Relaxing': 1, 
            'Exercising': 2,
            'Socializing': 3,
            'Commuting': 4,
            'Traveling': 5, 
            'Gaming': 6 
        }

    weather_map = {
            'Clear': 0,
            'Cloudy': 1,
            'Disgusting': 2,
            'Gloomy': 3,
            'Night': 4,
            'Rainy': 5,
            'Snowy': 6,
            'Sunny': 7,
            'Windy': 8
        }
    weather_data = get_current_weather(api_key, city)
    weather_code = weather_map.get(weather_data, 0)  # Default to 'Clear' if not found
    activity_code = activity_map.get(activity, 0)
    age_code = get_age_code(profile.age)
    songs_df = pd.read_csv('/Users/tanzimfarhan/Desktop/FeelBeat/sub_main/FER_SystemModule/spotify_1M_mood3.csv')
    context_data = pd.DataFrame({
        'activity': [activity_code],
        'weather': [weather_code],
        'age': [age_code],
    })

    emotion, recommendations = hybrid_recommendation(mood,predictions,context_data,songs_df)
    return emotion, recommendations

    
    


def get_age_code(age):
    age_code = -1
    # Check which numeric code the age belongs to
    if age < 18:
        age_code = 0
    elif 18 <= age <= 24:
        age_code = 1
    elif 25 <= age <= 34:
        age_code = 2
    elif 35 <= age <= 44:
        age_code = 3
    elif 45 <= age <= 54:
        age_code = 4
    elif 55 <= age <= 64:
        age_code = 5
    else:
        age_code = 6
    
    return age_code



def get_current_weather(api_key, city):
    base_url = "http://api.weatherapi.com/v1/current.json"
    params = {
        'key': api_key,
        'q': city,
        'aqi': 'no'  # Air Quality Index is not needed, set to 'no'
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    return data


api_key = '908ba200454f4d82926144343241406' 
city = 'Dhaka'

