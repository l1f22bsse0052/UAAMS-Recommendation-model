from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load model
MODEL_PATH = '/home/zar/program_recommender/models/program_recommender_simple.pkl'

def load_model():
    """Load the pickled model"""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded successfully!")
        print(f"📊 Loaded {model['metadata']['total_programs']} programs")
        print(f"⚖️  Weighting: {model['metadata']['weighting']['fsc_weight']*100:.0f}% FSc, {model['metadata']['weighting']['matric_weight']*100:.0f}% Matric")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

MODEL = load_model()

def calculate_user_weighted_score(matric_score, fsc_score):
    """Calculate user's weighted score using 60% FSc, 40% Matric"""
    return (0.6 * fsc_score) + (0.4 * matric_score)

def find_closest_programs(user_weighted_score, k=4):
    """Find programs with closest weighted scores to user's score"""
    if MODEL is None:
        raise ValueError("Model not loaded")
    
    results = []
    for prog in MODEL['programs']:
        # Calculate absolute difference
        difference = abs(user_weighted_score - prog['Weighted_Score_Needed'])
        
        # Calculate percentage difference
        percent_diff = (difference / prog['Weighted_Score_Needed']) * 100 if prog['Weighted_Score_Needed'] > 0 else 100
        
        results.append({
            'program': prog['CAMPUS|PROGRAM'],
            'closing_merit': round(prog['AGGR'], 2),
            'weighted_score_needed': round(prog['Weighted_Score_Needed'], 2),
            'difference': round(difference, 2),
            'percent_difference': round(percent_diff, 1),
            'formula': str(prog['FORMULA'])[:80] + '...' if prog['FORMULA'] and len(str(prog['FORMULA'])) > 80 else str(prog['FORMULA']),
            'source': prog['SOURCE_LINK'] if prog['SOURCE_LINK'] else 'Not available'
        })
    
    # Sort by difference (absolute difference)
    results.sort(key=lambda x: x['difference'])
    
    # Add chance prediction based on difference
    for r in results[:k]:
        if r['difference'] <= 2:
            r['chance'] = "Excellent - Very high chance"
            r['recommendation'] = "Strongly recommended - You have a great chance!"
        elif r['difference'] <= 5:
            r['chance'] = "Good - Strong chance"
            r['recommendation'] = "Recommended - Good match for your scores"
        elif r['difference'] <= 10:
            r['chance'] = "Moderate - Competitive"
            r['recommendation'] = "Consider - Need slight improvement in either subject"
        elif r['difference'] <= 15:
            r['chance'] = "Low - Need improvement"
            r['recommendation'] = "Challenging - Consider improving your scores"
        else:
            r['chance'] = "Very Low - Consider alternatives"
            r['recommendation'] = "Not recommended - Look for programs with lower requirements"
    
    return results[:k]

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'name': 'Program Recommendation API',
        'version': '2.0',
        'description': 'Uses 60% FSc + 40% Matric weighting',
        'formula': 'Your Score = (FSc × 0.6) + (Matric × 0.4)',
        'endpoints': {
            '/health': 'GET - Health check',
            '/recommend': 'POST - Get program recommendations',
            '/stats': 'GET - Model statistics',
            '/calculate': 'POST - Calculate your weighted score'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if MODEL is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    return jsonify({
        'status': 'healthy',
        'total_programs': MODEL['metadata']['total_programs'],
        'weighting': MODEL['metadata']['weighting']
    })

@app.route('/calculate', methods=['POST'])
def calculate_score():
    """Calculate user's weighted score without recommendations"""
    try:
        data = request.get_json()
        fsc = data.get('fsc')
        matric = data.get('matric')
        
        if fsc is None or matric is None:
            return jsonify({'error': 'Please provide both fsc and matric'}), 400
        
        if not (0 <= fsc <= 100) or not (0 <= matric <= 100):
            return jsonify({'error': 'Scores must be between 0 and 100'}), 400
        
        user_score = calculate_user_weighted_score(matric, fsc)
        
        return jsonify({
            'status': 'success',
            'your_marks': {'matric': matric, 'fsc': fsc},
            'weighted_score': round(user_score, 2),
            'formula': '60% × FSc + 40% × Matric',
            'interpretation': f'Your score of {round(user_score, 2)}% determines which programs you qualify for'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    """Get program recommendations based on weighted score"""
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
        
        if not (1 <= k <= 20):
            return jsonify({'error': 'k must be between 1 and 20'}), 400
        
        # Calculate user's weighted score
        user_weighted_score = calculate_user_weighted_score(matric, fsc)
        
        # Find closest programs
        recommendations = find_closest_programs(user_weighted_score, k)
        
        response = {
            'status': 'success',
            'user_input': {
                'matric': matric,
                'fsc': fsc,
                'your_weighted_score': round(user_weighted_score, 2)
            },
            'formula_used': 'Your Score = (FSc × 0.6) + (Matric × 0.4)',
            'recommendations': recommendations,
            'summary': {
                'total_recommendations': len(recommendations),
                'closest_program': recommendations[0]['program'] if recommendations else None,
                'closest_weighted_score_needed': recommendations[0]['weighted_score_needed'] if recommendations else None,
                'difference': recommendations[0]['difference'] if recommendations else None,
                'your_chance': recommendations[0]['chance'] if recommendations else None
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get model statistics"""
    if MODEL is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    programs = MODEL['programs']
    weighted_scores = [p['Weighted_Score_Needed'] for p in programs]
    aggr_values = [p['AGGR'] for p in programs]
    
    return jsonify({
        'status': 'success',
        'total_programs': len(programs),
        'weighting': MODEL['metadata']['weighting'],
        'statistics': {
            'weighted_scores': {
                'min': round(min(weighted_scores), 2),
                'max': round(max(weighted_scores), 2),
                'avg': round(np.mean(weighted_scores), 2),
                'std': round(np.std(weighted_scores), 2)
            },
            'closing_merits': {
                'min': round(min(aggr_values), 2),
                'max': round(max(aggr_values), 2),
                'avg': round(np.mean(aggr_values), 2),
                'std': round(np.std(aggr_values), 2)
            }
        }
    })

if __name__ == '__main__':
    print("="*60)
    print("🚀 STARTING RECOMMENDATION SERVER")
    print("Formula: Your Score = (FSc × 0.6) + (Matric × 0.4)")
    print("="*60)
    
    if MODEL:
        print("\n📡 Starting server on http://localhost:5000")
        print("\n🔗 Available endpoints:")
        print("   POST /calculate  - Calculate your weighted score")
        print("   POST /recommend  - Get program recommendations")
        print("   GET  /stats      - Model statistics")
        print("   GET  /health     - Health check")
        print("\n💡 Example request:")
        print('   curl -X POST http://localhost:5000/recommend \\')
        print('        -H "Content-Type: application/json" \\')
        print('        -d \'{"fsc": 75, "matric": 80}\'')
        print("\n" + "="*60)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to load model. Please run training script first!")