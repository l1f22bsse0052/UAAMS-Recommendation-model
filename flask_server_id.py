from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = '/home/zar/program_recommender/models/program_recommender_id.pkl'

def load_model():
    """Load the pickled model"""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded successfully!")
        print(f"📊 Loaded {model['metadata']['total_programs']} programs")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

MODEL = load_model()

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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if MODEL is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    return jsonify({
        'status': 'healthy',
        'total_programs': MODEL['metadata']['total_programs'],
        'version': MODEL['metadata']['version']
    })

if __name__ == '__main__':
    print("="*60)
    print("🚀 STARTING RECOMMENDATION API SERVER")
    print("Formula: Test% = 0.7×FSc + 0.3×Matric")
    print("Then: Predicted AGGR using each university's weights")
    print("="*60)
    
    if MODEL:
        print(f"\n📊 Loaded {MODEL['metadata']['total_programs']} programs")
        print("\n📡 Server running on http://localhost:5000")
        print("\n🔗 Endpoints:")
        print("   POST /recommend        - Get program recommendations")
        print("   POST /calculate_aggr   - Calculate AGGR for specific program")
        print("   GET  /health           - Health check")
        print("\n💡 Example:")
        print('   curl -X POST http://localhost:5000/recommend \\')
        print('        -H "Content-Type: application/json" \\')
        print('        -d \'{"fsc": 63, "matric": 70}\'')
        print("\n" + "="*60)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to load model. Please run training first!")