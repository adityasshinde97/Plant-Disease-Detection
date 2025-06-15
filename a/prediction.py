import streamlit as st
import tensorflow as tf
import numpy as np

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    confidence = prediction[0][result_index]
    return result_index, confidence  # return both class and confidence

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Home Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "C:/Users/ASUS/OneDrive/Desktop/Plant Dieseas Prediction/home_page.jpeg"
    st.image(image_path,use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
""")
    
#About Page
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
    #### Welcome to the Future of Plant Care!

Is your plant looking a little off? Don’t worry—we’ve got you covered! Our Plant Disease Detection System is here to help you identify and diagnose plant diseases in seconds with the power of AI and Computer Vision.

Simply upload a photo of your plant, and our intelligent system will instantly analyze it, providing you with an accurate diagnosis and actionable insights to help your plant thrive. No more guessing—just quick, reliable results at your fingertips!

Why Choose Us?

* Real-Time Results: Snap a picture and receive an instant diagnosis, along with possible disease causes and treatments.
* Smart AI Diagnosis: Our advanced machine learning models have been trained on thousands of plant images, offering you the best chance for an accurate diagnosis.
* Wide Coverage: From your home garden to large-scale farming, we detect diseases in a broad range of plants.
* Simple & User-Friendly: Designed for anyone—from beginners to experts—our easy-to-use interface makes plant care a breeze.
* Helpful Guidance: Along with the diagnosis, get expert advice on how to treat or prevent the disease, ensuring your plant stays healthy and vibrant.

Don’t wait for your plants to suffer—take control of their health today! Just upload a picture and let us do the rest.
     Content
    1. Train (70295 images)
    2. Valid (17572 image)
    3. Test (33 images)
""")


#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
   # Disease info: symptoms, prevention, and pesticide recommendation
disease_info = {
    'Apple___Apple_scab': {
        'symptoms': 'Olive-green or brown spots on leaves, fruit, and twigs.',
        'prevention': 'Use resistant varieties, remove infected leaves, and apply fungicide.',
        'pesticide': 'Apply Captan or Mancozeb-based fungicides during early spring.'
    },
    'Apple___Black_rot': {
        'symptoms': 'Dark, sunken lesions on fruit and leaves.',
        'prevention': 'Prune affected branches, remove mummified fruit, and apply fungicides.',
        'pesticide': 'Use fungicides like thiophanate-methyl or myclobutanil.'
    },
    'Apple___Cedar_apple_rust': {
        'symptoms': 'Orange or rust-colored spots on leaves.',
        'prevention': 'Remove nearby cedar trees and apply appropriate fungicides.',
        'pesticide': 'Apply fungicides such as mancozeb or myclobutanil during bud break.'
    },
    'Apple___healthy': {
        'symptoms': 'No visible disease symptoms.',
        'prevention': 'Maintain good plant hygiene and regular monitoring.',
        'pesticide': 'No pesticide needed. Monitor regularly.'
    },
    'Blueberry___healthy': {
        'symptoms': 'No visible disease symptoms.',
        'prevention': 'Ensure proper soil drainage and routine inspection.',
        'pesticide': 'No pesticide needed. Focus on good cultural practices.'
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'symptoms': 'White, powdery fungal growth on leaves and fruit.',
        'prevention': 'Improve air circulation by pruning and avoid overhead watering.',
        'pesticide': 'Apply sulfur-based fungicides or potassium bicarbonate.'
    },
    'Cherry_(including_sour)___healthy': {
        'symptoms': 'No visible disease symptoms.',
        'prevention': 'Regular pruning, monitoring, and balanced fertilization.',
        'pesticide': 'No pesticide needed. Maintain cleanliness and good airflow.'
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'symptoms': 'Rectangular gray or tan lesions on lower leaves that progress upward.',
        'prevention': 'Rotate crops and use resistant hybrids.',
        'pesticide': 'Use fungicides like azoxystrobin or propiconazole during tasseling.'
    },
    'Corn_(maize)___Common_rust_': {
        'symptoms': 'Reddish-brown pustules scattered over leaves.',
        'prevention': 'Use resistant corn varieties and ensure crop rotation.',
        'pesticide': 'Apply fungicides such as tebuconazole or pyraclostrobin at early infection.'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'symptoms': 'Cigar-shaped gray-green lesions on lower leaves.',
        'prevention': 'Plant resistant hybrids and avoid dense planting.',
        'pesticide': 'Use fungicides like chlorothalonil or mancozeb if detected early.'
    },
    'Corn_(maize)___healthy': {
        'symptoms': 'No visible disease symptoms.',
        'prevention': 'Maintain optimal soil conditions and monitor for early signs of disease.',
        'pesticide': 'No pesticide required. Implement preventive agronomic practices.'
    },
    'Grape___Black_rot': {
        'symptoms': 'Small, circular black spots on leaves, brown rotting spots on fruit.',
        'prevention': 'Prune affected areas, remove mummified berries, and ensure good air circulation.',
        'pesticide': 'Apply fungicides like myclobutanil or mancozeb during early growth stages.'
    },
    'Grape___Esca_(Black_Measles)': {
        'symptoms': 'Interveinal yellowing on leaves, dark streaks in wood, and berry spotting or shriveling.',
        'prevention': 'Avoid excessive pruning wounds, disinfect tools, and remove infected vines.',
        'pesticide': 'No specific curative pesticide. Apply protective fungicides like trifloxystrobin early in the season.'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'symptoms': 'Irregular brown spots with yellow halos on leaves, premature leaf drop.',
        'prevention': 'Ensure proper pruning and remove infected leaves from vineyard.',
        'pesticide': 'Use protective fungicides such as chlorothalonil or copper-based sprays.'
    },
    'Grape___healthy': {
        'symptoms': 'No visible disease symptoms.',
        'prevention': 'Regularly monitor the vineyard and maintain good sanitation practices.',
        'pesticide': 'No pesticide required. Preventive management is sufficient.'
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'symptoms': 'Yellowing of leaves, misshapen fruit with bitter taste, and overall tree decline.',
        'prevention': 'Control psyllid population, use certified disease-free planting material, and remove infected trees.',
        'pesticide': 'Apply insecticides like imidacloprid to control Asian citrus psyllid (ACP).'
    },
    'Peach___Bacterial_spot': {
        'symptoms': 'Small, dark spots on leaves and fruit, often leading to defoliation and blemished peaches.',
        'prevention': 'Use resistant varieties, avoid overhead watering, and prune infected branches.',
        'pesticide': 'Apply copper-based bactericides during early leaf development.'
    },
    'Peach___healthy': {
        'symptoms': 'No visible disease symptoms.',
        'prevention': 'Use resistant cultivars and maintain proper pruning and hygiene.',
        'pesticide': 'No pesticide required. Follow preventive care and seasonal monitoring.'
    },
    'Pepper,_bell___Bacterial_spot': {
        'symptoms': 'Water-soaked spots on leaves and fruit that may turn brown and scabby.',
        'prevention': 'Avoid overhead irrigation, use disease-free seeds, and rotate crops.',
        'pesticide': 'Apply copper-based sprays or fixed copper fungicides.'
    },
    'Pepper,_bell___healthy': {
        'symptoms': 'No visible disease symptoms.',
        'prevention': 'Maintain clean soil and tools, and avoid high humidity around the plant.',
        'pesticide': 'No pesticide necessary. Maintain routine crop care.'
    },
    'Potato___Early_blight': {
        'symptoms': 'Brown spots with concentric rings on older leaves, yellowing, and leaf drop.',
        'prevention': 'Rotate crops, avoid overhead watering, and space plants adequately.',
        'pesticide': 'Use fungicides such as chlorothalonil or azoxystrobin at first sign of disease.'
    },
    'Potato___Late_blight': {
        'symptoms': 'Dark, water-soaked lesions on leaves and stems, white fungal growth under leaves.',
        'prevention': 'Use resistant varieties, destroy infected plants, and avoid overhead irrigation.',
        'pesticide': 'Apply fungicides like metalaxyl or mancozeb preventively or at early signs.'
    },
    'Potato___healthy': {
        'symptoms': 'No visible disease symptoms.',
        'prevention': 'Practice crop rotation, ensure proper drainage, and monitor regularly.',
        'pesticide': 'No pesticide needed unless disease signs appear.'
    },
    'Raspberry___healthy': {
        'symptoms': 'No visible disease symptoms.',
        'prevention': 'Ensure proper spacing, pruning, and sanitation to avoid fungal issues.',
        'pesticide': 'No pesticide needed. Regular monitoring and maintenance is sufficient.'
    },
    'Soybean___healthy': {
        'symptoms': 'No visible disease symptoms.',
        'prevention': 'Use certified seeds, rotate crops, and manage weeds properly.',
        'pesticide': 'No pesticide required. Continue good agricultural practices.'
    },
    'Squash___Powdery_mildew': {
        'symptoms': 'White, powdery spots on both upper and lower leaf surfaces, stems, and flowers.',
        'prevention': 'Provide good air circulation and avoid overhead watering.',
        'pesticide': 'Use fungicides such as neem oil, sulfur, or potassium bicarbonate.'
    },
    'Strawberry___Leaf_scorch': {
        'symptoms': 'Purplish spots on leaves that enlarge and merge, leading to scorched appearance and defoliation.',
        'prevention': 'Avoid overhead irrigation, remove infected leaves, and improve air circulation.',
        'pesticide': 'Apply fungicides like myclobutanil or captan during early growing stages.'
    },
    'Strawberry___healthy': {
        'symptoms': 'No visible disease symptoms.',
        'prevention': 'Maintain proper spacing, clean mulch, and regular inspection for early signs.',
        'pesticide': 'No pesticide needed. Focus on preventive care.'
    },
     'Tomato___Bacterial_spot': {
        'symptoms': 'Small, dark, water-soaked spots on leaves and fruits that enlarge and may cause yellowing or fruit lesions.',
        'prevention': 'Use certified seeds, avoid overhead watering, and practice crop rotation.',
        'pesticide': 'Apply copper-based bactericides or fixed copper sprays regularly during wet conditions.'
    },
    'Tomato___Early_blight': {
        'symptoms': 'Dark, concentric-ringed spots on older leaves, followed by yellowing and defoliation.',
        'prevention': 'Remove crop debris, rotate crops, and use resistant varieties.',
        'pesticide': 'Use fungicides such as chlorothalonil or mancozeb at first appearance of symptoms.'
    },
    'Tomato___Late_blight': {
        'symptoms': 'Large, irregular, water-soaked lesions with white mold on undersides of leaves, affecting stems and fruit.',
        'prevention': 'Destroy infected plants, avoid overhead irrigation, and ensure good airflow.',
        'pesticide': 'Use systemic fungicides like metalaxyl or fluopicolide at first sign of infection.'
    },
    'Tomato___Leaf_Mold': {
        'symptoms': 'Yellow spots on upper leaf surface with olive-green to gray mold on the underside.',
        'prevention': 'Ensure good ventilation and avoid high humidity in greenhouses.',
        'pesticide': 'Apply fungicides such as chlorothalonil or copper-based sprays.'
    },
    'Tomato___Septoria_leaf_spot': {
        'symptoms': 'Small, circular spots with dark borders and gray centers on lower leaves.',
        'prevention': 'Remove infected leaves, avoid overhead watering, and space plants well.',
        'pesticide': 'Use fungicides like mancozeb or chlorothalonil at regular intervals.'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'symptoms': 'Tiny yellow or white spots on leaves, fine webbing, and eventual leaf bronzing and drop.',
        'prevention': 'Maintain humidity, use insecticidal soap, and control weeds.',
        'pesticide': 'Apply miticides like abamectin or use horticultural oils for control.'
    },
    'Tomato___Target_Spot': {
        'symptoms': 'Brown, circular lesions with concentric rings and yellow halos on leaves and fruit.',
        'prevention': 'Practice crop rotation, improve air circulation, and remove infected debris.',
        'pesticide': 'Use fungicides like azoxystrobin or pyraclostrobin for effective control.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'symptoms': 'Upward curling of leaves, yellowing, stunted growth, and flower drop.',
        'prevention': 'Control whiteflies, use resistant varieties, and remove infected plants.',
        'pesticide': 'Use insecticides like imidacloprid to control whitefly vectors.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'symptoms': 'Mottled light and dark green patterns on leaves, leaf curling, and distorted fruit.',
        'prevention': 'Disinfect tools, avoid handling plants after using tobacco, and remove infected plants.',
        'pesticide': 'No chemical cure. Use preventive measures and resistant varieties.'
    },
    'Tomato___healthy': {
        'symptoms': 'No visible disease symptoms.',
        'prevention': 'Use certified seeds, rotate crops, and maintain hygiene.',
        'pesticide': 'No pesticide required. Routine care and monitoring is recommended.'
    }
    # Continue for all other classes similarly...
}

if(st.button("Show Image")):
        st.image(test_image,use_container_width=True)

# Predict Button
if(st.button("Predict")):
    with st.spinner("Please Wait.."):
        st.write("Our Prediction")
        model = tf.keras.models.load_model('trained_model.keras')  # Load model once here
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to batch
        prediction = model.predict(input_arr)

        result_index = np.argmax(prediction)
        confidence = np.max(prediction)  # Get the maximum probability/confidence

        # Define Class
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
                      'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                      'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                      'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                      'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                      'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                      'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                      'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
                      'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
                      'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
                      'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                      'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                      'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

      # Set a confidence threshold (like 70%)
        threshold = 0.7

        if confidence < threshold:
            st.error("The uploaded image does not match any known plant disease with high confidence. Please upload a clear plant leaf image.")
        else:
            predicted_disease = class_name[result_index]
            st.success(f"Model Prediction: **{predicted_disease}**")

            # Show Symptoms and Prevention
            if predicted_disease in disease_info:
                st.subheader("Disease Details")
                st.markdown(f"**Symptoms:** {disease_info[predicted_disease]['symptoms']}")
                st.markdown(f"**Prevention:** {disease_info[predicted_disease]['prevention']}")
                st.markdown(f"**Pesticide Recommendation:** {disease_info[predicted_disease]['pesticide']}")
            else:
                st.info("Detailed information for this disease is not available.")