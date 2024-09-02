import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Load the model and dataset
@st.cache_data
def load_model_and_data():
    with open('disease_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    df = pd.read_csv('ber.csv')  # Load your dataset appropriately
    
    return model, df

model, df = load_model_and_data()

# Features and labels
X = df['Symptoms_List']
y = df['Disease']

# Symptom categories
body_regions = {
    'Digestive System': [
        'stomach_pain', 'acidity', 'vomiting', 'indigestion', 'constipation',
        'pain_during_bowel_movements', 'abdominal_pain', 'nausea',
        'loss_of_appetite', 'diarrhoea', 'spotting_urination', 'passage_of_gases',
        'yellowing_of_eyes', 'distention_of_abdomen', 'irritation_in_anus', 'excessive_hunger'
    ],
    'Respiratory System': [
        'cough', 'breathlessness', 'continuous_sneezing', 'high_fever'
    ],
    'Skin Symptoms': [
        'itching', 'skin_rash', 'patches_in_throat', 'pus_filled_pimples',
        'burning_micturition', 'skin_peeling', 'blister', 'yellow_crust_ooze',
        'blackheads', 'dischromic_patches', 'yellowish_skin', 'nodal_skin_eruptions',
        'watering_from_eyes'
    ],
    'Nervous System': [
        'headache', 'dizziness', 'fatigue', 'restlessness', 'loss_of_concentration',
        'blurred_and_distorted_vision', 'altered_sensorium', 'loss_of_balance'
    ],
    'Musculoskeletal System': [
        'back_pain', 'joint_pain', 'muscle_weakness', 'neck_pain', 'muscle_wasting',
        'stiff_neck', 'pain_in_anal_region', 'swollen_legs', 'painful_walking',
        'hip_joint_pain', 'movement_stiffness', 'knee_pain'
    ],
    'Cardiovascular': [
        'high_fever', 'shivering', 'cold_hands_and_feets'
    ],
    'General Symptoms': [
        'weakness_in_limbs', 'weight_loss', 'weight_gain', 'fatigue', 'sweating',
        'anxiety', 'irritability', 'dehydration', 'weakness_of_one_body_side',
        'mood_swings', 'continuous_feel_of_urine', 'cramps', 'excessive_hunger'
    ],
    'Urinary Symptoms': [
        'burning_micturition', 'bladder_discomfort', 'foul_smell_of_urine',
        'spotting_urination', 'frequent_urination'
    ],
    'Other': [
        'extra_marital_contacts', 'family_history', 'irregular_sugar_level',
        'scarring', 'small_dents_in_nails', 'red_sore_around_nose', 'swollen_joints'
    ]
}

# Disease names for dropdown
disease_names = sorted(df['Disease'].unique())

# Function to select symptoms
def select_symptoms():
    selected_symptoms = []
    for category, symptoms in body_regions.items():
        st.subheader(category)
        selected = st.multiselect(f"Select symptoms from {category}", symptoms)
        selected_symptoms.extend(selected)
    return selected_symptoms

# Function to get recommendations based on disease name
def get_recommendations_by_disease(disease_name):
    disease_info = df[df['Disease'] == disease_name]
    
    if not disease_info.empty:
        medications = disease_info['Medication'].values[0]
        precautions = disease_info['Precautions_List'].values[0]
        workouts = disease_info['Workout'].values[0]
        diet = disease_info['Diet'].values[0]
    else:
        medications = precautions = workouts = diet = "No information available"
    
    return {
        'Disease': disease_name,
        'Medications': medications,
        'Precautions': precautions,
        'Workouts': workouts,
        'Diet': diet
    }

# Predict top diseases with probabilities
def predict_top_diseases_with_probs(symptoms, top_n=5):
    symptom_vector = model.named_steps['tfidfvectorizer'].transform([' '.join(symptoms)])
    probas = model.named_steps['multinomialnb'].predict_proba(symptom_vector)
    
    diseases = model.classes_
    probas = probas[0]
    
    disease_probas = list(zip(diseases, probas))
    sorted_diseases = sorted(disease_probas, key=lambda x: x[1], reverse=True)
    
    top_diseases = sorted_diseases[:top_n]
    
    # Convert probabilities to percentage and create DataFrame
    df_probas = pd.DataFrame(top_diseases, columns=['Disease', 'Probability'])
    df_probas['Probability'] = df_probas['Probability'] * 100  # Express as percentage
    return df_probas

# Get recommendations table
def get_recommendations_table(diseases):
    recommendations = []
    for disease in diseases:
        disease_info = get_recommendations_by_disease(disease)
        recommendations.append(disease_info)
    
    df_recommendations = pd.DataFrame(recommendations)
    return df_recommendations

# Streamlit interface
st.title('Disease Prediction and Recommendations')

# Select method
option = st.selectbox(
    'How would you like to get recommendations?',
    ['Select Symptoms', 'Enter or Select Disease Name']
)

if option == 'Select Symptoms':
    user_symptoms = select_symptoms()

    if st.button('Predict'):
        if user_symptoms:
            top_diseases_df = predict_top_diseases_with_probs(user_symptoms)
            st.subheader('Top Diseases and Probabilities')
            st.dataframe(top_diseases_df)
            
            # Create a selector for detailed information
            selected_disease = st.selectbox("Select a disease to get detailed information:", top_diseases_df['Disease'])
            
            if selected_disease:
                disease_info = get_recommendations_by_disease(selected_disease)
                st.subheader(f"Detailed Information for {selected_disease}")
                st.write(f"**Medications:** {disease_info['Medications']}")
                st.write(f"**Precautions:** {disease_info['Precautions']}")
                st.write(f"**Workouts:** {disease_info['Workouts']}")
                st.write(f"**Diet:** {disease_info['Diet']}")
            
            # Create recommendations table
            recommendations_df = get_recommendations_table(top_diseases_df['Disease'])
            st.subheader('Recommendations for Diseases')
            st.dataframe(recommendations_df)
        else:
            st.warning('Please select symptoms.')

elif option == 'Enter or Select Disease Name':
    # Dropdown for disease names
    disease_name = st.selectbox('Select a disease:', disease_names)

    if st.button('Get Recommendations'):
        if disease_name:
            disease_info = get_recommendations_by_disease(disease_name)
            st.subheader(f"Recommendations for {disease_name}")
            st.write(f"**Medications:** {disease_info['Medications']}")
            st.write(f"**Precautions:** {disease_info['Precautions']}")
            st.write(f"**Workouts:** {disease_info['Workouts']}")
            st.write(f"**Diet:** {disease_info['Diet']}")
        else:
            st.warning('Please select a disease.')
