from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, make_response
from pymongo import MongoClient
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os
from dotenv import load_dotenv
import random
import bcrypt
from datetime import datetime, timedelta
from functools import wraps
import json
import requests
from bson import ObjectId
import pickle
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import secrets

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.permanent_session_lifetime = timedelta(days=1)

# MongoDB setup
client = MongoClient(os.getenv('MONGO_URI', 'mongodb://localhost:27017/'))
db = client.college_predictor


# Load trained model and encoders
def load_model():
    """Load trained model and encoders, train if not exists"""
    try:
        # Create models/saved directory if it doesn't exist
        os.makedirs('models/saved', exist_ok=True)

        model_path = 'models/saved/model.pkl'

        # If model doesn't exist, train it
        if not os.path.exists(model_path):
            print("🔄 No trained model found. Training model...")
            from models.train_models import train_enhanced_model
            train_enhanced_model()
            print("✅ Model training completed!")

        # Load the model
        model = pickle.load(open(model_path, 'rb'))
        le_state = pickle.load(open('models/saved/le_state.pkl', 'rb'))
        le_exam = pickle.load(open('models/saved/le_exam.pkl', 'rb'))
        le_category = pickle.load(open('models/saved/le_category.pkl', 'rb'))
        le_type = pickle.load(open('models/saved/le_type.pkl', 'rb'))
        return model, le_state, le_exam, le_category, le_type

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None, None, None


# Global variables for model
model, le_state, le_exam, le_category, le_type = load_model()


def init_db():
    """Initialize database with comprehensive data from CSV files"""
    try:
        # Create collections if they don't exist
        if 'users' not in db.list_collection_names():
            db.create_collection('users')
            print("✅ Users collection created")

        if 'colleges' not in db.list_collection_names():
            db.create_collection('colleges')
            print("✅ Colleges collection created")
            load_college_data_from_csv()

        if 'email_verifications' not in db.list_collection_names():
            db.create_collection('email_verifications')
            print("✅ Email verifications collection created")

        print("✅ Database initialized successfully")

    except Exception as e:
        print(f"❌ Database initialization error: {e}")


def load_college_data_from_csv():
    """Load college data from CSV files"""
    try:
        # Load engineering colleges data
        eng_data = []
        with open('data/engineering_colleges.csv', 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                parts = line.strip().split(',')
                if len(parts) >= 8:
                    eng_data.append({
                        'college_id': int(parts[0]),
                        'name': parts[1],
                        'state': parts[2],
                        'exam_type': parts[3],
                        'category': parts[4],
                        'cutoff_rank': int(parts[5]),
                        'marks_cutoff': int(parts[6]),
                        'website': parts[7],
                        'type': 'Engineering'
                    })

        # Load medical colleges data
        med_data = []
        with open('data/medical_colleges.csv', 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                parts = line.strip().split(',')
                if len(parts) >= 8:
                    med_data.append({
                        'college_id': int(parts[0]),
                        'name': parts[1],
                        'state': parts[2],
                        'exam_type': parts[3],
                        'category': parts[4],
                        'cutoff_rank': int(parts[5]),
                        'marks_cutoff': int(parts[6]),
                        'website': parts[7],
                        'type': 'Medical'
                    })

        # Insert all data
        all_colleges = eng_data + med_data
        if all_colleges:
            db.colleges.insert_many(all_colleges)
            print(f"✅ Inserted {len(all_colleges)} colleges from CSV files")
        else:
            print("❌ No college data found in CSV files")

    except Exception as e:
        print(f"❌ Error loading CSV data: {e}")


def login_required(f):
    """Decorator to require login for routes"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


def no_cache(f):
    """Decorator to prevent caching for secure pages"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        response = make_response(f(*args, **kwargs))
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response

    return decorated_function


def email_verified_required(f):
    """Decorator to require email verification"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first', 'error')
            return redirect(url_for('login'))

        user = db.users.find_one({'_id': ObjectId(session['user_id'])})
        if not user or not user.get('email_verified', False):
            flash('Please verify your email address first', 'error')
            return redirect(url_for('verify_email_page'))
        return f(*args, **kwargs)

    return decorated_function


@app.before_request
def initialize_data_on_first_request():
    """Initialize database on first request"""
    init_db()


def send_verification_email(email, otp):
    """Send verification email with OTP"""
    try:
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', 587))
        sender_email = os.getenv('EMAIL_USER')
        sender_password = os.getenv('EMAIL_PASSWORD')

        if not all([smtp_server, sender_email, sender_password]):
            print("❌ Email configuration missing")
            return False

        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = email
        message['Subject'] = 'College Predictor - Email Verification'

        body = f"""
        <html>
        <body>
            <h2>College Predictor - Email Verification</h2>
            <p>Thank you for registering with College Predictor!</p>
            <p>Your verification code is: <strong>{otp}</strong></p>
            <p>Enter this code on the verification page to complete your registration.</p>
            <p>This code will expire in 10 minutes.</p>
            <br>
            <p>Best regards,<br>College Predictor Team</p>
        </body>
        </html>
        """

        message.attach(MIMEText(body, 'html'))

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(message)

        print(f"✅ Verification email sent to {email}")
        return True

    except Exception as e:
        print(f"❌ Error sending verification email: {e}")
        return False


def get_gemini_response(user_message):
    """Get response from Google Gemini using REST API"""
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("❌ Gemini API key not found")
            return None

        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={api_key}"

        prompt = f"""You are an expert AI assistant for Indian college admissions. Specialize in:
        - Engineering admissions: JEE Main, JEE Advanced, state CET exams (KCET, MHT CET, etc.)
        - Medical admissions: NEET exam
        - College predictions, cutoff ranks, admission procedures
        - Reservation categories: General, OBC, SC, ST

        User Question: {user_message}

        Provide helpful, accurate information about Indian college admissions. Keep responses concise and informative (max 150 words)."""

        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "maxOutputTokens": 300,
                "temperature": 0.7
            }
        }

        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.post(url, json=payload, headers=headers, timeout=30)

        if response.status_code == 200:
            data = response.json()
            if 'candidates' in data and len(data['candidates']) > 0:
                return data['candidates'][0]['content']['parts'][0]['text']
            else:
                print("❌ No candidates in Gemini response")
                return None
        else:
            print(f"❌ Gemini API Error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        return None


def get_chatbot_response(user_message, context=None):
    """Enhanced chatbot with interactive prediction capability using trained model"""
    user_message = user_message.lower().strip()

    # Prediction context handling
    if context and context.get('awaiting_prediction'):
        return handle_prediction_input(user_message, context)

    # Quick options for predictions
    prediction_keywords = {
        'engineering': ['engineering', 'jee', 'iit', 'nit', 'btech', 'predict engineering', 'engineering prediction'],
        'medical': ['medical', 'neet', 'mbbs', 'doctor', 'predict medical', 'medical prediction']
    }

    for field, keywords in prediction_keywords.items():
        if any(keyword in user_message for keyword in keywords):
            return start_prediction_flow(field)

    # Greetings and basic queries
    if any(word in user_message for word in ['hello', 'hi', 'hey', 'namaste']):
        return {
            'response': "Hello! I'm your college admission assistant. I can help with:\n• Engineering (JEE, CET) predictions\n• Medical (NEET) predictions\n• Cutoffs and admission procedures\n\nWhat would you like to explore?",
            'options': ['Engineering Prediction', 'Medical Prediction', 'JEE Main Info', 'NEET Info']
        }

    if any(word in user_message for word in ['thank', 'thanks']):
        return {'response': "You're welcome! Feel free to ask more questions. Good luck with your preparations! 🎓"}

    if any(word in user_message for word in ['bye', 'goodbye']):
        return {
            'response': "Goodbye! Best wishes for your college admissions journey. Come back if you have more questions!"}

    # Default responses for unknown queries
    default_responses = [
        "I specialize in Indian college admissions. Ask me about Engineering (JEE, CET) or Medical (NEET) predictions!",
        "I can help you with college predictions based on your rank and marks. Try 'engineering prediction' or 'medical prediction'!",
        "For personalized college recommendations, use the prediction forms or ask me to start a prediction!"
    ]

    return {'response': random.choice(default_responses)}


def start_prediction_flow(field):
    """Start interactive prediction flow"""
    if field == 'engineering':
        return {
            'response': "🚀 Great! Let's predict engineering colleges. I'll need some details:\n\n1. 📝 Exam Type (JEE Main, CET, etc.)\n2. 🏆 Your Rank\n3. 📊 12th Percentage\n4. 👥 Category\n5. 🗺️ Preferred State\n\nPlease provide your exam type:",
            'context': {'field': 'engineering', 'step': 'exam_type', 'awaiting_prediction': True},
            'options': ['JEE Main', 'CET', 'COMEDK', 'VITEEE']
        }
    else:  # medical
        return {
            'response': "🚀 Great! Let's predict medical colleges. I'll need some details:\n\n1. 🏆 Your NEET Rank\n2. 📊 12th PCB Percentage\n3. 👥 Category\n4. 🗺️ Preferred State\n\nPlease provide your NEET rank:",
            'context': {'field': 'medical', 'step': 'rank', 'awaiting_prediction': True}
        }


def handle_prediction_input(user_message, context):
    """Handle step-by-step prediction input using trained model"""
    field = context['field']
    step = context['step']

    try:
        if step == 'exam_type' and field == 'engineering':
            valid_exams = ['jee main', 'cet', 'comedk', 'viteee', 'bitsat']
            exam_mapping = {
                'jee main': 'JEE Main',
                'cet': 'CET',
                'comedk': 'COMEDK',
                'viteee': 'VITEEE',
                'bitsat': 'BITSAT'
            }

            if user_message.lower() not in valid_exams:
                return {
                    'response': "❌ Please choose a valid exam type: JEE Main, CET, COMEDK, VITEEE, or BITSAT",
                    'context': context,
                    'options': ['JEE Main', 'CET', 'COMEDK', 'VITEEE']
                }

            context['exam_type'] = exam_mapping[user_message.lower()]
            context['step'] = 'rank'
            return {
                'response': f"✅ Got it! Exam Type: {context['exam_type']}\n\nNow please provide your rank:",
                'context': context
            }

        elif step == 'rank':
            rank = int(user_message)
            if rank <= 0:
                return {
                    'response': "❌ Please provide a valid positive rank:",
                    'context': context
                }
            context['rank'] = rank
            context['step'] = 'marks'
            marks_label = "12th PCB Percentage" if field == 'medical' else "12th Percentage"
            return {
                'response': f"✅ Rank: {rank}\n\nNow please provide your {marks_label}:",
                'context': context
            }

        elif step == 'marks':
            marks = float(user_message)
            if marks < 0 or marks > 100:
                return {
                    'response': "❌ Please provide a valid percentage (0-100):",
                    'context': context
                }
            context['marks_12th'] = marks
            context['step'] = 'category'
            return {
                'response': f"✅ Marks: {marks}%\n\nNow please provide your category:",
                'context': context,
                'options': ['General', 'OBC', 'SC', 'ST']
            }

        elif step == 'category':
            valid_categories = ['general', 'obc', 'sc', 'st']
            if user_message.lower() not in valid_categories:
                return {
                    'response': "❌ Please choose a valid category: General, OBC, SC, or ST",
                    'context': context,
                    'options': ['General', 'OBC', 'SC', 'ST']
                }
            context['category'] = user_message.title()
            context['step'] = 'state'
            return {
                'response': f"✅ Category: {context['category']}\n\nNow please provide your preferred state:",
                'context': context,
                'options': ['All India', 'Karnataka', 'Maharashtra', 'Tamil Nadu', 'Delhi', 'Uttar Pradesh']
            }

        elif step == 'state':
            context['state'] = user_message.title()

            # Perform prediction using trained model
            if field == 'engineering':
                predictions = predict_colleges_with_model('engineering', context.get('exam_type', 'JEE Main'),
                                                          context['rank'], context['marks_12th'],
                                                          context['category'], context['state'])
            else:
                predictions = predict_colleges_with_model('medical', 'NEET', context['rank'],
                                                          context['marks_12th'], context['category'],
                                                          context['state'])

            # Format response
            if predictions:
                response_text = f"🎓 **Here are your predicted {field.title()} colleges:**\n\n"
                for i, college in enumerate(predictions[:5], 1):
                    response_text += f"**{i}. {college['name']}**\n"
                    response_text += f"   📍 State: {college['state']}\n"
                    response_text += f"   📊 Probability: {college['probability']}%\n"
                    response_text += f"   🏆 Cutoff Rank: {college['cutoff_rank']}\n"
                    response_text += f"   📝 Marks Cutoff: {college['marks_cutoff']}%\n\n"

                response_text += "💡 **Tip:** Higher probability means better chances based on your rank and marks."
            else:
                response_text = "❌ No suitable colleges found with your criteria. Try adjusting your rank, marks, or preferences."

            return {
                'response': response_text,
                'context': {'awaiting_prediction': False},
                'predictions': predictions
            }

    except ValueError:
        return {
            'response': "❌ Please provide a valid number:",
            'context': context
        }

    return {'response': "❌ I didn't understand that. Please try again.", 'context': context}


def predict_colleges_with_model(field, exam_type, rank, marks_12th, category, state):
    """Predict colleges using trained ML model with improved accuracy"""
    if model is None:
        return predict_colleges_fallback(field, exam_type, rank, marks_12th, category, state)

    try:
        # Prepare features for prediction
        features = prepare_features(field, exam_type, rank, marks_12th, category, state)
        if features is None:
            return predict_colleges_fallback(field, exam_type, rank, marks_12th, category, state)

        # Get all colleges for the field
        query = {'type': field.capitalize()}
        if state != 'All India':
            query['state'] = state
        if category:
            query['category'] = category

        colleges = list(db.colleges.find(query))

        suitable_colleges = []
        for college in colleges:
            # Update features with college-specific data
            features['cutoff_rank'] = college['cutoff_rank']
            features['marks_cutoff'] = college['marks_cutoff']

            # Create feature array for prediction
            feature_array = create_feature_array(features)

            # Predict probability using the model
            predicted_rank = model.predict(feature_array)[0]

            # Calculate probability based on predicted vs actual rank with improved algorithm
            probability = calculate_improved_probability(rank, college['cutoff_rank'], marks_12th,
                                                         college['marks_cutoff'])

            if probability > 5:  # Include more colleges with reasonable probability
                suitable_colleges.append({
                    'name': college['name'],
                    'state': college['state'],
                    'cutoff_rank': college['cutoff_rank'],
                    'marks_cutoff': college['marks_cutoff'],
                    'website': college['website'],
                    'probability': round(probability, 2)
                })

        # Sort by probability and return top results
        suitable_colleges.sort(key=lambda x: x['probability'], reverse=True)
        return suitable_colleges[:15]  # Return more colleges

    except Exception as e:
        print(f"❌ Model prediction error: {e}")
        return predict_colleges_fallback(field, exam_type, rank, marks_12th, category, state)


def calculate_improved_probability(student_rank, cutoff_rank, student_marks, cutoff_marks):
    """Improved probability calculation with better matching"""
    try:
        # Rank-based probability with wider range
        rank_ratio = student_rank / cutoff_rank if cutoff_rank > 0 else 1

        if rank_ratio <= 0.5:
            rank_prob = 90 + (1 - rank_ratio) * 10
        elif rank_ratio <= 1.0:
            rank_prob = 70 + (1 - rank_ratio) * 20
        elif rank_ratio <= 2.0:
            rank_prob = 40 + (2 - rank_ratio) * 30
        elif rank_ratio <= 3.0:
            rank_prob = 20 + (3 - rank_ratio) * 20
        else:
            rank_prob = max(5, 20 - (rank_ratio - 3) * 5)

        # Marks-based adjustment
        marks_ratio = student_marks / cutoff_marks if cutoff_marks > 0 else 1
        if marks_ratio >= 1.2:
            marks_bonus = 15
        elif marks_ratio >= 1.1:
            marks_bonus = 10
        elif marks_ratio >= 1.0:
            marks_bonus = 5
        elif marks_ratio >= 0.9:
            marks_bonus = 0
        elif marks_ratio >= 0.8:
            marks_bonus = -5
        else:
            marks_bonus = -10

        # Combined probability with adjusted weights
        probability = (rank_prob * 0.8) + (marks_bonus * 0.2)
        return max(1, min(99, probability))

    except Exception as e:
        print(f"❌ Error calculating probability: {e}")
        return 30  # Default probability


def prepare_features(field, exam_type, rank, marks_12th, category, state):
    """Prepare features for model prediction"""
    try:
        features = {
            'field': field,
            'exam_type': exam_type,
            'rank': rank,
            'marks_12th': marks_12th,
            'category': category,
            'state': state,
            'tier': 1 if rank <= 1000 else (2 if rank <= 10000 else 3),
            'state_importance': 1 if state in ['Delhi', 'Maharashtra', 'Karnataka', 'Tamil Nadu'] else 0.7
        }
        return features
    except Exception as e:
        print(f"❌ Error preparing features: {e}")
        return None


def create_feature_array(features):
    """Create feature array for model prediction"""
    try:
        # Encode categorical features with fallback
        state_encoded = le_state.transform([features['state']])[0] if features['state'] in le_state.classes_ else 0
        exam_encoded = le_exam.transform([features['exam_type']])[0] if features['exam_type'] in le_exam.classes_ else 0
        category_encoded = le_category.transform([features['category']])[0] if features[
                                                                                   'category'] in le_category.classes_ else 0
        type_encoded = le_type.transform([features['field']])[0] if features['field'] in le_type.classes_ else 0

        feature_array = np.array([[
            state_encoded, exam_encoded, category_encoded, type_encoded,
            features['marks_12th'], features['tier'], features['state_importance'],
            features.get('cutoff_rank', 0), features.get('marks_cutoff', 0)
        ]])
        return feature_array
    except Exception as e:
        print(f"❌ Error creating feature array: {e}")
        return np.array([[0, 0, 0, 0, features['marks_12th'], features['tier'], features['state_importance'], 0, 0]])


def predict_colleges_fallback(field, exam_type, rank, marks_12th, category, state):
    """Improved fallback prediction when model is not available"""
    query = {
        'type': field.capitalize(),
        'category': category
    }

    if state != 'All India':
        query['state'] = state

    colleges = list(db.colleges.find(query))

    suitable_colleges = []
    for college in colleges:
        probability = calculate_improved_probability(rank, college['cutoff_rank'], marks_12th, college['marks_cutoff'])

        if probability > 5:  # Lower threshold to include more colleges
            suitable_colleges.append({
                'name': college['name'],
                'state': college['state'],
                'cutoff_rank': college['cutoff_rank'],
                'marks_cutoff': college['marks_cutoff'],
                'website': college['website'],
                'probability': round(probability, 2)
            })

    suitable_colleges.sort(key=lambda x: x['probability'], reverse=True)
    return suitable_colleges[:15]


def get_ai_response(user_message, context=None):
    """Get AI response - try Gemini first, then fallback to enhanced local"""
    # Try Gemini API first
    gemini_response = get_gemini_response(user_message)
    if gemini_response:
        return {'response': gemini_response}

    # Fallback to enhanced local responses
    return get_chatbot_response(user_message, context)


def validate_email(email):
    """Validate email format"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_marks(marks):
    """Validate marks percentage"""
    try:
        marks = float(marks)
        return 0 <= marks <= 100
    except:
        return False


def validate_rank(rank):
    """Validate rank"""
    try:
        rank = int(rank)
        return rank > 0
    except:
        return False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
@no_cache
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        try:
            email = request.form['email']
            password = request.form['password']

            if not validate_email(email):
                return render_template('login.html', error='Invalid email format')

            user = db.users.find_one({'email': email})
            if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
                if not user.get('email_verified', False):
                    session['pending_user_id'] = str(user['_id'])
                    return redirect(url_for('verify_email_page'))

                session.permanent = True
                session['user_id'] = str(user['_id'])
                session['email'] = user['email']
                session['name'] = user['name']
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))
            else:
                return render_template('login.html', error='Invalid email or password')

        except Exception as e:
            print(f"Login error: {e}")
            return render_template('login.html', error='An error occurred during login')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
@no_cache
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        try:
            name = request.form['name']
            email = request.form['email']
            password = request.form['password']
            phone = request.form['phone']
            state = request.form['state']
            education = request.form['education']
            interests = request.form.getlist('interests')

            if not validate_email(email):
                return render_template('register.html', error='Invalid email format')

            if len(password) < 6:
                return render_template('register.html', error='Password must be at least 6 characters')

            # Check if user exists
            if db.users.find_one({'email': email}):
                return render_template('register.html', error='Email already exists')

            # Hash password
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

            # Generate OTP
            otp = str(secrets.randbelow(900000) + 100000)

            # Store OTP in verification collection
            db.email_verifications.insert_one({
                'email': email,
                'otp': otp,
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(minutes=10)
            })

            # Insert new user (not verified yet)
            user_id = db.users.insert_one({
                'name': name,
                'email': email,
                'password': hashed_password,
                'phone': phone,
                'state': state,
                'education': education,
                'interests': interests,
                'email_verified': False,
                'created_at': datetime.now()
            }).inserted_id

            print(f"✅ User inserted with ID: {user_id}")
            print(f"✅ User data: {db.users.find_one({'_id': user_id})}")

            # Send verification email
            if send_verification_email(email, otp):
                session['pending_user_id'] = str(user_id)
                # CHANGED: Don't show success message yet, just redirect to verification
                return redirect(url_for('verify_email_page'))
            else:
                # Clean up if email sending fails
                db.users.delete_one({'_id': user_id})
                db.email_verifications.delete_one({'email': email})
                return render_template('register.html', error='Failed to send verification email. Please try again.')

        except Exception as e:
            print(f"Registration error: {e}")
            return render_template('register.html', error='An error occurred during registration')

    return render_template('register.html')


@app.route('/verify-email', methods=['GET', 'POST'])
@no_cache
def verify_email_page():
    if 'pending_user_id' not in session:
        flash('Please register first', 'error')
        return redirect(url_for('register'))

    if request.method == 'POST':
        try:
            otp = request.form['otp']
            user_id = session['pending_user_id']

            user = db.users.find_one({'_id': ObjectId(user_id)})
            if not user:
                flash('User not found', 'error')
                return redirect(url_for('register'))

            # Check OTP
            verification = db.email_verifications.find_one({
                'email': user['email'],
                'otp': otp,
                'expires_at': {'$gt': datetime.now()}
            })

            if verification:
                # Mark email as verified
                db.users.update_one(
                    {'_id': ObjectId(user_id)},
                    {'$set': {'email_verified': True}}
                )

                # Clean up verification record
                db.email_verifications.delete_one({'_id': verification['_id']})

                # Set session
                session.permanent = True
                session['user_id'] = user_id
                session['email'] = user['email']
                session['name'] = user['name']
                session.pop('pending_user_id', None)

                flash('Email verified successfully!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid or expired verification code', 'error')
                return render_template('verify_email.html')

        except Exception as e:
            print(f"Verification error: {e}")
            flash('An error occurred during verification', 'error')
            return render_template('verify_email.html')

    return render_template('verify_email.html')


@app.route('/resend-otp')
@no_cache
def resend_otp():
    if 'pending_user_id' not in session:
        flash('Please register first', 'error')
        return redirect(url_for('register'))

    try:
        user = db.users.find_one({'_id': ObjectId(session['pending_user_id'])})
        if not user:
            flash('User not found', 'error')
            return redirect(url_for('register'))

        # Generate new OTP
        otp = str(secrets.randbelow(900000) + 100000)

        # Update verification record
        db.email_verifications.update_one(
            {'email': user['email']},
            {'$set': {
                'otp': otp,
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(minutes=10)
            }},
            upsert=True
        )

        if send_verification_email(user['email'], otp):
            flash('New verification code sent to your email', 'success')
        else:
            flash('Failed to send verification email', 'error')

        return redirect(url_for('verify_email_page'))

    except Exception as e:
        print(f"Resend OTP error: {e}")
        flash('An error occurred', 'error')
        return redirect(url_for('verify_email_page'))


@app.route('/dashboard')
@login_required
@email_verified_required
@no_cache
def dashboard():
    user = db.users.find_one({'_id': ObjectId(session['user_id'])})
    return render_template('index.html', username=session.get('name'), user=user)


@app.route('/engineering', methods=['GET', 'POST'])
@login_required
@email_verified_required
def engineering():
    if request.method == 'POST':
        exam_type = request.form['exam_type']
        rank = request.form['rank']
        marks_12th = request.form['marks_12th']
        category = request.form['category']
        state = request.form['state']

        # Validate inputs
        if not validate_rank(rank):
            flash('Invalid rank', 'error')
            return render_template('engineering.html')

        if not validate_marks(marks_12th):
            flash('Invalid marks percentage', 'error')
            return render_template('engineering.html')

        # Get predictions using trained model
        predictions = predict_colleges_with_model('engineering', exam_type, int(rank), float(marks_12th), category,
                                                  state)

        return render_template('results.html',
                               predictions=predictions,
                               field='Engineering',
                               exam_type=exam_type,
                               rank=rank,
                               marks=marks_12th)

    return render_template('engineering.html')


@app.route('/medical', methods=['GET', 'POST'])
@login_required
@email_verified_required
def medical():
    if request.method == 'POST':
        rank = request.form['rank']
        marks_12th = request.form['marks_12th']
        category = request.form['category']
        state = request.form['state']

        # Validate inputs
        if not validate_rank(rank):
            flash('Invalid rank', 'error')
            return render_template('medical.html')

        if not validate_marks(marks_12th):
            flash('Invalid marks percentage', 'error')
            return render_template('medical.html')

        # Get predictions using trained model
        predictions = predict_colleges_with_model('medical', 'NEET', int(rank), float(marks_12th), category, state)

        return render_template('results.html',
                               predictions=predictions,
                               field='Medical',
                               exam_type='NEET',
                               rank=rank,
                               marks=marks_12th)

    return render_template('medical.html')


@app.route('/chatbot')
@login_required
@email_verified_required
def chatbot_page():
    return render_template('chatbot.html')


@app.route('/chatbot', methods=['POST'])
@login_required
@email_verified_required
def chatbot_api():
    if request.method == 'POST':
        user_message = request.json.get('message')
        context = request.json.get('context')

        bot_response = get_ai_response(user_message, context)

        return jsonify(bot_response)


@app.route('/reset-chat', methods=['POST'])
@login_required
@email_verified_required
def reset_chat():
    """Reset chatbot conversation"""
    return jsonify({'status': 'success', 'message': 'Chat reset successfully'})


@app.route('/logout')
@no_cache
def logout():
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)