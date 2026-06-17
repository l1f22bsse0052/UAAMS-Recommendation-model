from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
import sys
import pandas as pd
import json
from datetime import datetime
import threading
import shutil
import re
from pathlib import Path

app = Flask(__name__)
CORS(app)

# ==================== CONFIGURATION ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'program_recommender_id.pkl')
EXCEL_PATH = os.path.join(BASE_DIR, 'data', 'Book1(1).xlsx')
BACKUP_DIR = os.path.join(BASE_DIR, 'models', 'backups')

# Ensure directories exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(EXCEL_PATH), exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

# Global variables
MODEL = None
IS_TRAINING = False
TRAINING_LOCK = threading.Lock()

# ==================== MODEL LOADING & TRAINING ====================

def load_model():
    """Load the pickled model, or train if it doesn't exist"""
    global MODEL
    
    # Check if model exists
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                MODEL = pickle.load(f)
            print(f"✅ Model loaded successfully!")
            print(f"📊 Loaded {MODEL['metadata']['total_programs']} programs")
            return MODEL
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            MODEL = None
            return None
    else:
        print(f"⚠️  Model file not found at: {MODEL_PATH}")
        
        # Check if Excel file exists
        if os.path.exists(EXCEL_PATH):
            print(f"📂 Found Excel file at: {EXCEL_PATH}")
            print("🔨 Attempting to train model...")
            
            try:
                # Import training function
                import importlib.util
                train_script = os.path.join(BASE_DIR, 'train_model_id.py')
                
                if os.path.exists(train_script):
                    # Import the training module
                    spec = importlib.util.spec_from_file_location("train_model_id", train_script)
                    train_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(train_module)
                    
                    # Train the model
                    success = train_module.train_model(EXCEL_PATH, MODEL_PATH)
                    
                    if success:
                        # Load the trained model
                        with open(MODEL_PATH, 'rb') as f:
                            MODEL = pickle.load(f)
                        print(f"✅ Model trained and loaded successfully!")
                        print(f"📊 Loaded {MODEL['metadata']['total_programs']} programs")
                        return MODEL
                    else:
                        print("❌ Training failed!")
                        MODEL = None
                        return None
                else:
                    print(f"❌ Training script not found at: {train_script}")
                    MODEL = None
                    return None
                    
            except Exception as e:
                print(f"❌ Error during training: {e}")
                MODEL = None
                return None
        else:
            print(f"❌ Excel file not found at: {EXCEL_PATH}")
            print("📝 Please ensure both files exist:")
            print(f"   1. Excel data: {EXCEL_PATH}")
            print(f"   2. Training script: {os.path.join(BASE_DIR, 'train_model_id.py')}")
            MODEL = None
            return None

def extract_weights(formula):
    """Extract matric, inter, and test weights from formula string"""
    if pd.isna(formula):
        return 0.17, 0.50, 0.33
    
    formula_str = str(formula)
    matric_match = re.search(r'matric\s*/\s*matricTotal\)\s*\*\s*([\d.]+)', formula_str)
    inter_match = re.search(r'inter\s*/\s*interTotal\)\s*\*\s*([\d.]+)', formula_str)
    test_match = re.search(r'test\s*/\s*400\)\s*\*\s*([\d.]+)', formula_str)
    
    w_matric = float(matric_match.group(1)) / 100 if matric_match else 0.17
    w_inter = float(inter_match.group(1)) / 100 if inter_match else 0.50
    w_test = float(test_match.group(1)) / 100 if test_match else 0.33
    
    return w_matric, w_inter, w_test

def calculate_weighted_score_needed(aggr, w_matric, w_fsc, w_test, assumed_test_pct=0.50):
    """Calculate what weighted score would be needed to achieve the closing AGGR"""
    test_contribution = assumed_test_pct * 100 * w_test
    remaining = aggr - test_contribution
    
    if remaining <= 0:
        return aggr
    
    if (w_matric + w_fsc) > 0:
        estimated_equal_score = remaining / (w_matric + w_fsc)
    else:
        estimated_equal_score = 50
    
    np.random.seed(abs(int(aggr * 100)) % 1000)
    variation = np.random.uniform(0.96, 1.04)
    weighted_score = estimated_equal_score * variation
    weighted_score = np.clip(weighted_score, 30, 100)
    
    return weighted_score

def train_model_from_data(df, model_path=MODEL_PATH):
    """Train model from DataFrame"""
    try:
        print("🔍 Extracting formula weights...")
        df['W_Matric'], df['W_FSc'], df['W_Test'] = zip(*df['FORMULA'].apply(extract_weights))
        
        print("📊 Calculating weighted scores...")
        np.random.seed(42)
        df['Weighted_Score_Needed'] = df.apply(
            lambda row: calculate_weighted_score_needed(
                row['AGGR'], row['W_Matric'], row['W_FSc'], row['W_Test']
            ), axis=1
        )
        
        # Prepare model data
        model_data = {
            'programs': [],
            'metadata': {
                'total_programs': len(df),
                'weighting': {'fsc_weight': 0.6, 'matric_weight': 0.4},
                'version': '2.0',
                'description': 'Simple weighted score comparison with MongoDB IDs',
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # Build programs list
        programs_without_ids = 0
        for _, row in df.iterrows():
            program_info = {
                'program': str(row['CAMPUS|PROGRAM']),
                'campus': str(row['CAMPUS']) if pd.notna(row.get('CAMPUS')) else None,
                'program_name': str(row['PROGRAM']) if pd.notna(row.get('PROGRAM')) else None,
                'closing_merit': round(row['AGGR'], 2),
                'weighted_score_needed': round(row['Weighted_Score_Needed'], 2),
                'mongo_id': str(row['MONGO_ID']) if pd.notna(row.get('MONGO_ID')) else None,
                'university_id': str(row['UNIVERSITY_ID']) if pd.notna(row.get('UNIVERSITY_ID')) else None,
                'formula': str(row['FORMULA']) if pd.notna(row.get('FORMULA')) else None,
                'source_link': str(row['SOURCE_LINK']) if pd.notna(row.get('SOURCE_LINK')) else None,
                'weights': {
                    'matric': round(row['W_Matric'], 3),
                    'fsc': round(row['W_FSc'], 3),
                    'test': round(row['W_Test'], 3)
                }
            }
            
            if program_info['mongo_id'] is None or program_info['university_id'] is None:
                programs_without_ids += 1
                
            model_data['programs'].append(program_info)
        
        # Save to pickle file
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved to {model_path}")
        print(f"📊 Model contains {len(df)} programs")
        
        return model_data
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        raise e

def load_excel_data(excel_path):
    """Load Excel data from file"""
    print(f"📂 Loading Excel file from {excel_path}...")
    
    try:
        df = pd.read_excel(excel_path, sheet_name='data (2)')
        print(f"✅ Loaded Sheet 'data (2)' with {len(df)} records")
    except:
        df = pd.read_excel(excel_path, sheet_name='data')
        print(f"✅ Loaded Sheet 'data' with {len(df)} records")
    
    # Clean data
    df = df.dropna(subset=['AGGR', 'CAMPUS|PROGRAM'])
    df = df[df['AGGR'] > 0]
    df = df.reset_index(drop=True)
    
    # Ensure required columns exist
    if 'MONGO_ID' not in df.columns:
        df['MONGO_ID'] = None
        print("⚠️  MONGO_ID column not found, setting to None")
    
    if 'UNIVERSITY_ID' not in df.columns:
        df['UNIVERSITY_ID'] = None
        print("⚠️  UNIVERSITY_ID column not found, setting to None")
    
    if 'FORMULA' not in df.columns:
        df['FORMULA'] = None
        print("⚠️  FORMULA column not found, setting to None")
    
    if 'SOURCE_LINK' not in df.columns:
        df['SOURCE_LINK'] = None
        print("⚠️  SOURCE_LINK column not found, setting to None")
    
    if 'CAMPUS' not in df.columns:
        df['CAMPUS'] = df['CAMPUS|PROGRAM'].apply(lambda x: x.split('|')[0] if '|' in str(x) else None)
        print("✅ Created CAMPUS column from CAMPUS|PROGRAM")
    
    if 'PROGRAM' not in df.columns:
        df['PROGRAM'] = df['CAMPUS|PROGRAM'].apply(lambda x: x.split('|')[1] if '|' in str(x) else None)
        print("✅ Created PROGRAM column from CAMPUS|PROGRAM")
    
    print(f"✅ Total valid programs: {len(df)}")
    return df

def retrain_model():
    """Retrain the model with current Excel data"""
    global IS_TRAINING, MODEL
    
    with TRAINING_LOCK:
        if IS_TRAINING:
            print("⚠️ Training already in progress, skipping...")
            return
        
        IS_TRAINING = True
        print(f"\n🔄 Starting model retraining at {datetime.now()}")
    
    try:
        # Create backup of current model
        if os.path.exists(MODEL_PATH):
            backup_name = f"program_recommender_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            backup_path = os.path.join(BACKUP_DIR, backup_name)
            shutil.copy2(MODEL_PATH, backup_path)
            print(f"📦 Created backup: {backup_name}")
        
        # Load data from Excel
        df = load_excel_data(EXCEL_PATH)
        
        # Train new model
        model_data = train_model_from_data(df, MODEL_PATH)
        
        # Update global model
        MODEL = model_data
        
        print(f"✅ Model retraining completed successfully at {datetime.now()}")
        print(f"📊 Total programs in model: {len(model_data['programs'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during retraining: {e}")
        # Attempt to restore from backup
        try:
            backups = sorted([f for f in os.listdir(BACKUP_DIR) if f.endswith('.pkl')])
            if backups:
                latest_backup = os.path.join(BACKUP_DIR, backups[-1])
                shutil.copy2(latest_backup, MODEL_PATH)
                print(f"🔄 Restored from backup: {backups[-1]}")
                MODEL = load_model()
        except Exception as restore_error:
            print(f"❌ Failed to restore backup: {restore_error}")
            MODEL = None
        
        return False
        
    finally:
        IS_TRAINING = False

def add_program_to_excel(program_data):
    """Add a new program to the Excel file"""
    try:
        # Load existing data
        if os.path.exists(EXCEL_PATH):
            df = pd.read_excel(EXCEL_PATH, sheet_name='data (2)')
        else:
            # Create new DataFrame with proper columns
            df = pd.DataFrame(columns=['AGGR', 'CAMPUS|PROGRAM', 'FORMULA', 'SOURCE_LINK', 'MONGO_ID', 'UNIVERSITY_ID', 'CAMPUS', 'PROGRAM'])
        
        # Prepare new row
        new_row = {
            'AGGR': float(program_data['closing_merit']),
            'CAMPUS|PROGRAM': f"{program_data['campus']}|{program_data['program_name']}",
            'FORMULA': program_data.get('formula', 'result = ((matric / matricTotal) * 17) + ((inter / interTotal) * 50) + ((test / 400) * 33)'),
            'SOURCE_LINK': program_data.get('source_link', ''),
            'MONGO_ID': program_data['mongo_id'],
            'UNIVERSITY_ID': program_data['university_id'],
            'CAMPUS': program_data['campus'],
            'PROGRAM': program_data['program_name']
        }
        
        # Add to DataFrame
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save to Excel
        with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl', mode='w') as writer:
            df.to_excel(writer, sheet_name='data (2)', index=False)
        
        print(f"✅ Program added to Excel: {new_row['CAMPUS|PROGRAM']}")
        return True
        
    except Exception as e:
        print(f"❌ Error adding program to Excel: {e}")
        return False

# ==================== RECOMMENDATION ENGINE ====================

def estimate_test_score(matric_pct, fsc_pct):
    """Estimate entry test score based on board marks"""
    return (0.7 * fsc_pct) + (0.3 * matric_pct)

def calculate_predicted_aggr(matric_pct, fsc_pct, weights):
    """Calculate predicted aggregate using university's formula"""
    test_pct = estimate_test_score(matric_pct, fsc_pct)
    
    predicted_aggr = (
        (matric_pct * weights['matric']) + 
        (fsc_pct * weights['fsc']) + 
        (test_pct * weights['test'])
    )
    
    return predicted_aggr, test_pct

def find_closest_programs(matric_pct, fsc_pct, k=4):
    """Find programs where predicted AGGR is closest to closing merit"""
    if MODEL is None:
        raise ValueError("Model not loaded")
    
    results = []
    
    for prog in MODEL['programs']:
        # Calculate predicted AGGR for this program
        predicted_aggr, estimated_test = calculate_predicted_aggr(
            matric_pct, fsc_pct, prog['weights']
        )
        
        # Calculate difference from actual closing merit
        difference = abs(predicted_aggr - prog['closing_merit'])
        percent_diff = (difference / prog['closing_merit']) * 100 if prog['closing_merit'] > 0 else 100
        
        results.append({
            'program': prog['program'],
            'campus': prog['campus'],
            'program_name': prog['program_name'],
            'closing_merit': prog['closing_merit'],
            'predicted_aggr': round(predicted_aggr, 2),
            'estimated_test_score': round(estimated_test, 2),
            'difference': round(difference, 2),
            'percent_difference': round(percent_diff, 1),
            'mongo_id': prog['mongo_id'],
            'university_id': prog['university_id'],
            'source_link': prog['source_link'],
            'weights': prog['weights']
        })
    
    # Sort by difference (smallest first)
    results.sort(key=lambda x: x['difference'])
    
    # Add chance prediction
    for r in results[:k]:
        if r['difference'] <= 2:
            r['chance'] = "Excellent"
            r['recommendation_text'] = "Strongly recommended - Your predicted aggregate meets/exceeds closing merit!"
            r['color_code'] = "#4CAF50"
        elif r['difference'] <= 5:
            r['chance'] = "Good"
            r['recommendation_text'] = "Recommended - Very close to closing merit"
            r['color_code'] = "#8BC34A"
        elif r['difference'] <= 10:
            r['chance'] = "Moderate"
            r['recommendation_text'] = "Competitive - Slightly below closing merit"
            r['color_code'] = "#FFC107"
        elif r['difference'] <= 15:
            r['chance'] = "Low"
            r['recommendation_text'] = "Challenging - Consider improving your scores"
            r['color_code'] = "#FF9800"
        else:
            r['chance'] = "Very Low"
            r['recommendation_text'] = "Not recommended - Look for programs with lower requirements"
            r['color_code'] = "#F44336"
    
    return results[:k]

# ==================== API ENDPOINTS ====================

@app.route('/recommend', methods=['POST'])
def recommend():
    """Get program recommendations based on direct AGGR prediction"""
    try:
        if MODEL is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        fsc = data.get('fsc')
        matric = data.get('matric')
        k = data.get('k', 4)
        
        if fsc is None or matric is None:
            return jsonify({'error': 'Please provide both "fsc" and "matric"'}), 400
        
        if not (0 <= fsc <= 100) or not (0 <= matric <= 100):
            return jsonify({'error': 'Scores must be between 0 and 100'}), 400
        
        # Calculate estimated test score
        estimated_test = estimate_test_score(matric, fsc)
        
        # Find closest programs
        recommendations = find_closest_programs(matric, fsc, k)
        
        response = {
            'status': 'success',
            'user_input': {
                'matric': matric,
                'fsc': fsc,
                'estimated_test_score': round(estimated_test, 2)
            },
            'estimation_formula': 'Test Score = (0.7 × FSc) + (0.3 × Matric)',
            'recommendations': recommendations,
            'summary': {
                'total_recommendations': len(recommendations),
                'closest_program': recommendations[0]['program'] if recommendations else None,
                'closest_mongo_id': recommendations[0]['mongo_id'] if recommendations else None,
                'predicted_aggr': recommendations[0]['predicted_aggr'] if recommendations else None,
                'closing_merit': recommendations[0]['closing_merit'] if recommendations else None
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/calculate_aggr', methods=['POST'])
def calculate_aggr():
    """Calculate predicted aggregate for specific program"""
    try:
        if MODEL is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        fsc = data.get('fsc')
        matric = data.get('matric')
        mongo_id = data.get('mongo_id')
        
        if not all([fsc, matric, mongo_id]):
            return jsonify({'error': 'Please provide fsc, matric, and mongo_id'}), 400
        
        # Find program
        program = None
        for prog in MODEL['programs']:
            if prog['mongo_id'] == mongo_id:
                program = prog
                break
        
        if not program:
            return jsonify({'error': 'Program not found'}), 404
        
        # Calculate predicted aggregate
        predicted_aggr, estimated_test = calculate_predicted_aggr(matric, fsc, program['weights'])
        
        return jsonify({
            'status': 'success',
            'program': program['program'],
            'closing_merit': program['closing_merit'],
            'predicted_aggr': round(predicted_aggr, 2),
            'estimated_test_score': round(estimated_test, 2),
            'difference': round(predicted_aggr - program['closing_merit'], 2),
            'weights_used': program['weights']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/add-program', methods=['POST'])
def add_program():
    """Add a new program to the dataset and retrain the model."""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['campus', 'program_name', 'closing_merit', 'mongo_id', 'university_id']
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Validate closing_merit
        try:
            closing_merit = float(data['closing_merit'])
            if not (0 <= closing_merit <= 100):
                return jsonify({'error': 'closing_merit must be between 0 and 100'}), 400
        except ValueError:
            return jsonify({'error': 'closing_merit must be a number'}), 400
        
        # Check if program already exists
        existing_program = None
        if MODEL and 'programs' in MODEL:
            for prog in MODEL['programs']:
                if prog.get('mongo_id') == data['mongo_id']:
                    existing_program = prog
                    break
        
        if existing_program:
            return jsonify({
                'error': f'Program with mongo_id {data["mongo_id"]} already exists',
                'existing_program': existing_program
            }), 409
        
        # Add program to Excel
        success = add_program_to_excel(data)
        
        if not success:
            return jsonify({'error': 'Failed to add program to dataset'}), 500
        
        # Start retraining in background
        thread = threading.Thread(target=retrain_model)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'Program added successfully. Model retraining started in background.',
            'program_added': {
                'campus': data['campus'],
                'program_name': data['program_name'],
                'closing_merit': data['closing_merit'],
                'mongo_id': data['mongo_id'],
                'university_id': data['university_id']
            },
            'retraining_status': 'started'
        }), 202
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/add-programs-bulk', methods=['POST'])
def add_programs_bulk():
    """Add multiple programs at once."""
    try:
        data = request.get_json()
        
        if not isinstance(data, list):
            return jsonify({'error': 'Expected a list of programs'}), 400
        
        if len(data) == 0:
            return jsonify({'error': 'Empty list provided'}), 400
        
        added_count = 0
        failed_count = 0
        errors = []
        
        for idx, program in enumerate(data):
            try:
                # Validate required fields
                required_fields = ['campus', 'program_name', 'closing_merit', 'mongo_id', 'university_id']
                missing_fields = [f for f in required_fields if f not in program]
                
                if missing_fields:
                    errors.append(f"Program {idx}: Missing fields: {', '.join(missing_fields)}")
                    failed_count += 1
                    continue
                
                # Add program
                success = add_program_to_excel(program)
                if success:
                    added_count += 1
                else:
                    errors.append(f"Program {idx}: Failed to add")
                    failed_count += 1
                    
            except Exception as e:
                errors.append(f"Program {idx}: Error - {str(e)}")
                failed_count += 1
        
        # Start retraining if any programs were added
        if added_count > 0:
            thread = threading.Thread(target=retrain_model)
            thread.daemon = True
            thread.start()
        
        return jsonify({
            'status': 'success',
            'message': f'Added {added_count} programs, {failed_count} failed',
            'added_count': added_count,
            'failed_count': failed_count,
            'errors': errors if errors else None,
            'retraining_status': 'started' if added_count > 0 else 'not_started'
        }), 202
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    """Manually trigger model retraining using the current Excel data."""
    global IS_TRAINING
    
    if IS_TRAINING:
        return jsonify({
            'status': 'error',
            'message': 'Training already in progress'
        }), 409
    
    # Start retraining in background
    thread = threading.Thread(target=retrain_model)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'status': 'success',
        'message': 'Model retraining started in background',
        'retraining_status': 'started'
    }), 202

@app.route('/training-status', methods=['GET'])
def training_status():
    """Check if model training is in progress"""
    global IS_TRAINING
    
    return jsonify({
        'is_training': IS_TRAINING,
        'model_loaded': MODEL is not None,
        'total_programs': MODEL['metadata']['total_programs'] if MODEL else 0,
        'last_training_date': MODEL['metadata']['training_date'] if MODEL else None
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get detailed information about the current model"""
    if MODEL is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Count programs with and without IDs
    programs_with_ids = sum(1 for p in MODEL['programs'] if p.get('mongo_id') and p.get('university_id'))
    
    return jsonify({
        'model_loaded': True,
        'metadata': MODEL['metadata'],
        'statistics': {
            'total_programs': len(MODEL['programs']),
            'programs_with_ids': programs_with_ids,
            'programs_without_ids': len(MODEL['programs']) - programs_with_ids
        },
        'sample_programs': MODEL['programs'][:5]
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if MODEL is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    
    return jsonify({
        'status': 'healthy',
        'total_programs': MODEL['metadata']['total_programs'],
        'version': MODEL['metadata']['version'],
        'training_date': MODEL['metadata']['training_date']
    })

@app.route('/backup', methods=['GET'])
def list_backups():
    """List all available model backups"""
    if not os.path.exists(BACKUP_DIR):
        return jsonify({'backups': []})
    
    backups = []
    for file in sorted(os.listdir(BACKUP_DIR), reverse=True):
        if file.endswith('.pkl'):
            file_path = os.path.join(BACKUP_DIR, file)
            backups.append({
                'filename': file,
                'size': os.path.getsize(file_path),
                'created': datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
            })
    
    return jsonify({
        'backups': backups,
        'total_backups': len(backups)
    })

@app.route('/restore/<backup_filename>', methods=['POST'])
def restore_backup(backup_filename):
    """Restore a model from backup"""
    global MODEL
    
    try:
        backup_path = os.path.join(BACKUP_DIR, backup_filename)
        
        if not os.path.exists(backup_path):
            return jsonify({'error': 'Backup file not found'}), 404
        
        # Create backup of current model before restoring
        if os.path.exists(MODEL_PATH):
            current_backup = f"pre_restore_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            shutil.copy2(MODEL_PATH, os.path.join(BACKUP_DIR, current_backup))
        
        # Restore from backup
        shutil.copy2(backup_path, MODEL_PATH)
        MODEL = load_model()
        
        return jsonify({
            'status': 'success',
            'message': f'Model restored from {backup_filename}',
            'total_programs': MODEL['metadata']['total_programs'] if MODEL else 0
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== LOAD MODEL ON STARTUP ====================
# This runs when Gunicorn starts the app
print("="*60)
print("🚀 STARTING RECOMMENDATION API SERVER WITH PIPELINE")
print("Formula: Test% = 0.7×FSc + 0.3×Matric")
print("Then: Predicted AGGR using each university's weights")
print("="*60)

# Load model - this will run for both Gunicorn and direct execution
MODEL = load_model()

if MODEL:
    print(f"\n📊 Loaded {MODEL['metadata']['total_programs']} programs")
    print(f"📡 Server running on port 10000 (Render) or 4000 (local)")
else:
    print("\n⚠️  Model not loaded. Please check the error messages above.")
    print("💡 The server will still run but /recommend will return errors.")

# ==================== MAIN (for local development) ====================

if __name__ == '__main__':
    print("\n🔗 Endpoints:")
    print("   POST /recommend        - Get program recommendations")
    print("   POST /calculate_aggr   - Calculate AGGR for specific program")
    print("   POST /add-program      - Add new program and retrain")
    print("   POST /add-programs-bulk - Add multiple programs")
    print("   POST /retrain          - Manually retrain model")
    print("   GET  /training-status  - Check training status")
    print("   GET  /model-info       - Get model information")
    print("   GET  /health           - Health check")
    print("   GET  /backup           - List available backups")
    print("   POST /restore/<file>   - Restore from backup")
    print("\n" + "="*60)
    
    app.run(debug=True, host='0.0.0.0', port=4000)
