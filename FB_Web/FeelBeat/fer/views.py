from django.shortcuts import render,redirect
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseBadRequest
import cv2 
import numpy as np
import base64
import cv2
from io import BytesIO
from PIL import Image
from .utils import process_image
from threading import Thread


# Create your views here.
@login_required

# def upload_and_process(request):
#     if request.method == 'POST':
#         # Convert base64 image to numpy array
#         image_data = request.POST.get('image_data')
#         format, imgstr = image_data.split(';base64,')
#         ext = format.split('/')[-1]
#         image = Image.open(BytesIO(base64.b64decode(imgstr)))
#         image_np = np.array(image.convert('RGB'))
#         image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

#         # Process image using the existing process_image function
#         emotion, recommended_songs = process_image(image_np)  # Assume process_image accepts numpy array

#         return render(request, 'fer/results.html', {
#             'emotion': emotion,
#             'recommended_songs': recommended_songs.to_dict(orient='records') if not recommended_songs.empty else []
#         })
#     else:
#         return render(request, 'fer/upload.html')

@login_required
# def upload_and_process(request):
#     if request.method == 'POST':
#         # Get image data and activity from POST request
#         image_data = request.POST.get('image_data')
#         activity = request.POST.get('activity', None)

#         # Ensure that image data is provided
#         if not image_data:
#             return HttpResponseBadRequest("No image data provided")

#         # Check if image data is in the expected base64 format
#         if ';base64,' not in image_data:
#             return HttpResponseBadRequest("Invalid image data format")

#         # Split the image_data into format and base64 string
#         format, imgstr = image_data.split(';base64,')
#         try:
#             # Convert base64 string to an image
#             image = Image.open(BytesIO(base64.b64decode(imgstr)))
#             image_np = np.array(image.convert('RGB'))
#             image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

#             # Process the image (assuming this function is defined elsewhere)
#             emotion, recommended_songs = process_image(image_np, activity)  # Process image accepts numpy array

#             # Check if the result contains recommended songs
#             if recommended_songs.empty:
#                 return render(request, 'fer/results.html', {
#                     'emotion': emotion,
#                     'recommended_songs': []
#                 })

#             return render(request, 'fer/results.html', {
#                 'emotion': emotion,
#                 'recommended_songs': recommended_songs.to_dict(orient='records')
#             })
#         except Exception as e:
#             return HttpResponseBadRequest(f"Error processing image: {e}")
#     else:
#         return render(request, 'fer/upload.html')

def upload_and_process(request):
    if request.method == 'POST':
        image_data = request.POST.get('image_data')
        activity = request.POST.get('activity', None)  # Ensure this is the correct type (str or int)

        if ';base64,' not in image_data:
            return HttpResponseBadRequest("Invalid image data format")

        format, imgstr = image_data.split(';base64,')
        try:
            image = Image.open(BytesIO(base64.b64decode(imgstr)))
            image_np = np.array(image.convert('RGB'))
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            if not isinstance(image_np, np.ndarray) or not isinstance(activity, (str, int)):
                return HttpResponseBadRequest("Invalid data types for image or activity")

            emotion, recommended_songs = process_image(image_np, activity)

            return render(request, 'fer/results.html', {
                'emotion': emotion,
                'recommended_songs': recommended_songs.to_dict(orient='records') if not recommended_songs.empty else []
            })
        except Exception as e:
            return HttpResponseBadRequest(f"Error processing image: {e}")
    else:
        return render(request, 'fer/upload.html')


    


