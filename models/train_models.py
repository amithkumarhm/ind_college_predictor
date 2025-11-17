import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import time


def generate_enhanced_sample_data():
    """Generate enhanced sample dataset with realistic patterns"""

    # Engineering colleges with realistic patterns
    engineering_colleges = []
    college_id = 1

    # IITs
    iits = [
        ('Indian Institute of Technology Bombay', 'Maharashtra', 100, 85),
        ('Indian Institute of Technology Delhi', 'Delhi', 500, 80),
        ('Indian Institute of Technology Madras', 'Tamil Nadu', 300, 82),
        ('Indian Institute of Technology Kanpur', 'Uttar Pradesh', 400, 81),
        ('Indian Institute of Technology Kharagpur', 'West Bengal', 600, 79),
    ]

    for name, state, base_rank, base_marks in iits:
        for category, rank_multiplier, marks_deduction in [('General', 1, 0), ('OBC', 5, -5), ('SC', 15, -10),
                                                           ('ST', 20, -13)]:
            engineering_colleges.append({
                'college_id': college_id,
                'name': name,
                'state': state,
                'exam_type': 'JEE Main',
                'category': category,
                'cutoff_rank': base_rank * rank_multiplier,
                'marks_cutoff': base_marks + marks_deduction,
                'website': f'https://www.{name.lower().replace(" ", "")}.ac.in',
                'type': 'Engineering'
            })
            college_id += 1

    # NITs
    nits = [
        ('National Institute of Technology Karnataka', 'Karnataka', 5000, 75),
        ('National Institute of Technology Trichy', 'Tamil Nadu', 4500, 76),
        ('National Institute of Technology Warangal', 'Telangana', 5500, 74),
        ('National Institute of Technology Surathkal', 'Karnataka', 6000, 73),
    ]

    for name, state, base_rank, base_marks in nits:
        for category, rank_multiplier, marks_deduction in [('General', 1, 0), ('OBC', 1.6, -5), ('SC', 3, -10),
                                                           ('ST', 3.6, -13)]:
            engineering_colleges.append({
                'college_id': college_id,
                'name': name,
                'state': state,
                'exam_type': 'JEE Main',
                'category': category,
                'cutoff_rank': int(base_rank * rank_multiplier),
                'marks_cutoff': base_marks + marks_deduction,
                'website': f'https://www.{name.lower().replace(" ", "")}.ac.in',
                'type': 'Engineering'
            })
            college_id += 1

    # Private colleges
    private_colleges = [
        ('RV College of Engineering', 'Karnataka', 'COMEDK', 12000, 68),
        ('BMS College of Engineering', 'Karnataka', 'COMEDK', 15000, 64),
        ('Vellore Institute of Technology', 'Tamil Nadu', 'VITEEE', 15000, 65),
        ('SRM Institute of Science and Technology', 'Tamil Nadu', 'SRMJEEE', 20000, 60),
    ]

    for name, state, exam, base_rank, base_marks in private_colleges:
        for category, rank_multiplier, marks_deduction in [('General', 1, 0), ('OBC', 1.4, -4), ('SC', 2, -8),
                                                           ('ST', 2.4, -10)]:
            engineering_colleges.append({
                'college_id': college_id,
                'name': name,
                'state': state,
                'exam_type': exam,
                'category': category,
                'cutoff_rank': int(base_rank * rank_multiplier),
                'marks_cutoff': base_marks + marks_deduction,
                'website': f'https://www.{name.lower().replace(" ", "")}.edu.in',
                'type': 'Engineering'
            })
            college_id += 1

    # Medical colleges
    medical_colleges = []

    # Top medical colleges
    medicals = [
        ('All India Institute of Medical Sciences Delhi', 'Delhi', 100, 85),
        ('Christian Medical College Vellore', 'Tamil Nadu', 500, 80),
        ('Armed Forces Medical College Pune', 'Maharashtra', 1000, 78),
        ('Maulana Azad Medical College Delhi', 'Delhi', 800, 79),
    ]

    for name, state, base_rank, base_marks in medicals:
        for category, rank_multiplier, marks_deduction in [('General', 1, 0), ('OBC', 5, -5), ('SC', 15, -10),
                                                           ('ST', 20, -13)]:
            medical_colleges.append({
                'college_id': college_id,
                'name': name,
                'state': state,
                'exam_type': 'NEET',
                'category': category,
                'cutoff_rank': base_rank * rank_multiplier,
                'marks_cutoff': base_marks + marks_deduction,
                'website': f'https://www.{name.lower().replace(" ", "")}.edu.in',
                'type': 'Medical'
            })
            college_id += 1

    return pd.DataFrame(engineering_colleges), pd.DataFrame(medical_colleges)


def train_enhanced_model():
    """Train enhanced prediction model with better features and 500+ epochs equivalent"""

    print("🔄 Starting enhanced model training...")
    start_time = time.time()

    # Generate enhanced sample data
    eng_df, med_df = generate_enhanced_sample_data()

    # Combine datasets
    df = pd.concat([eng_df, med_df], ignore_index=True)

    # Feature engineering
    df['tier'] = df['cutoff_rank'].apply(lambda x: 1 if x <= 1000 else (2 if x <= 10000 else 3))
    df['state_importance'] = df['state'].apply(
        lambda x: 1 if x in ['Delhi', 'Maharashtra', 'Karnataka', 'Tamil Nadu'] else 0.7)

    # Additional features for better prediction
    df['rank_to_cutoff_ratio'] = df['cutoff_rank'] / 10000  # Normalize rank
    df['marks_normalized'] = df['marks_cutoff'] / 100  # Normalize marks

    # Encode categorical variables
    le_state = LabelEncoder()
    le_exam = LabelEncoder()
    le_category = LabelEncoder()
    le_type = LabelEncoder()

    df['state_encoded'] = le_state.fit_transform(df['state'])
    df['exam_encoded'] = le_exam.fit_transform(df['exam_type'])
    df['category_encoded'] = le_category.fit_transform(df['category'])
    df['type_encoded'] = le_type.fit_transform(df['type'])

    # Enhanced features
    X = df[['state_encoded', 'exam_encoded', 'category_encoded', 'type_encoded',
            'marks_cutoff', 'tier', 'state_importance', 'rank_to_cutoff_ratio', 'marks_normalized']]
    y = df['cutoff_rank']

    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"📊 Dataset size: {len(df)} records")
    print(f"🎯 Training set: {len(X_train)} records")
    print(f"🧪 Test set: {len(X_test)} records")

    # Train enhanced model with more estimators (equivalent to 500 epochs for ensemble)
    model = RandomForestRegressor(
        n_estimators=500,  # Equivalent to 500 epochs for ensemble
        max_depth=25,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        verbose=1,
        n_jobs=-1  # Use all available cores
    )

    print("🔄 Training model with 500 estimators (equivalent to epochs)...")
    print("📈 Training progress:")

    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save model and encoders
    os.makedirs('models/saved', exist_ok=True)
    pickle.dump(model, open('models/saved/model.pkl', 'wb'))
    pickle.dump(le_state, open('models/saved/le_state.pkl', 'wb'))
    pickle.dump(le_exam, open('models/saved/le_exam.pkl', 'wb'))
    pickle.dump(le_category, open('models/saved/le_category.pkl', 'wb'))
    pickle.dump(le_type, open('models/saved/le_type.pkl', 'wb'))

    training_time = time.time() - start_time

    print("\n✅ Enhanced ML Model trained and saved successfully!")
    print(f"⏱️  Training time: {training_time:.2f} seconds")
    print(f"📊 Dataset size: {len(df)} records")
    print(f"🎯 Features used: {X.columns.tolist()}")
    print(f"🌳 Number of estimators (epochs): 500")
    print(f"📈 Mean Squared Error: {mse:.4f}")
    print(f"📊 R² Score: {r2:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n🔍 Feature Importance:")
    print(feature_importance)

    # Additional model insights
    print(f"\n📋 Model Parameters:")
    print(f"   - Max Depth: {model.max_depth}")
    print(f"   - Min Samples Split: {model.min_samples_split}")
    print(f"   - Min Samples Leaf: {model.min_samples_leaf}")
    print(f"   - Number of Features: {model.n_features_in_}")


if __name__ == '__main__':
    train_enhanced_model()