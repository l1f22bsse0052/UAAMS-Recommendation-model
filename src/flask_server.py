from flask import Flask, request, jsonify
import pickle
import numpy as np
from typing import Dict, List, Any

app = Flask(__name__)

# Global variables to hold the model
MODEL = None
MODEL_PATH = '/home/zar/program_recommender/models/program_recommender.pkl'

def load_model(model_path: str = MODEL_PATH):
    """Load the pickled model"""
    global MODEL
    try:
        with open(model_path, 'rb') as f:
            MODEL = pickle.load(f)
        print(f"✅ Model loaded successfully!")
        print(f"📊 Loaded {MODEL['metadata']['total_programs']} programs")
        return True
    except FileNotFoundError:
        print(f"❌ Model file {model_path} not found!")
        print("Please run train_model.py first")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def weighted_knn(matric_score: float, fsc_score: float, k: int = 4, 
                 w_matric: float = 0.3, w_fsc: float = 0.7) -> List[Dict]:
    """
    Perform weighted k-NN to find closest programs
    
    Args:
        matric_score: User's Matric percentage
        fsc_score: User's FSc percentage
        k: Number of neighbors to return
        w_matric: Weight for Matric (30%)
        w_fsc: Weight for FSc (70%)
    
    Returns:
        List of dictionaries with program recommendations
    """
    if MODEL is None:
        raise ValueError("Model not loaded")
    
    results = []
    programs = MODEL['programs']
    
    for prog in programs:
        # Calculate weighted Euclidean distance
        matric_diff = (matric_score - prog['Estimated_Matric_Pct']) ** 2
        fsc_diff = (fsc_score - prog['Estimated_FSc_Pct']) ** 2
        distance = np.sqrt(w_matric * matric_diff + w_fsc * fsc_diff)
        
        # Calculate user's estimated aggregate for this program's formula
        user_aggregate = (
            (matric_score * prog['W_Matric']) + 
            (fsc_score * prog['W_FSc']) + 
            (MODEL['metadata']['assumed_test_pct'] * 100 * prog['W_Test'])
        )
        
        results.append({
            'program': prog['CAMPUS|PROGRAM'],
            'closing_merit': round(prog['AGGR'], 2),
            'similarity_distance': round(distance, 3),
            'estimated_matric_needed': round(prog['Estimated_Matric_Pct'], 1),
            'estimated_fsc_needed': round(prog['Estimated_FSc_Pct'], 1),
            'your_estimated_aggregate': round(user_aggregate, 2),
            'gap_to_closing': round(prog['AGGR'] - user_aggregate, 2),
            'formula': prog['FORMULA'] if prog['FORMULA'] else 'Standard formula',
            'source': prog['SOURCE_LINK'] if prog['SOURCE_LINK'] else 'Not available'
        })
    
    # Sort by distance and return top k
    results.sort(key=lambda x: x['similarity_distance'])
    return results[:k]

def calculate_chance_percentage(gap: float) -> str:
    """Calculate admission chance based on gap to closing merit"""
    if gap <= -10:
        return "Low - Need significant improvement"
    elif gap <= -5:
        return "Below Average - Consider alternative programs"
    elif gap <= -2:
        return "Competitive - Close range"
    elif gap <= 0:
        return "Good - In the competitive range"
    elif gap <= 5:
        return "Very Good - Strong chance"
    else:
        return "Excellent - Very high chance"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'total_programs': MODEL['metadata']['total_programs'] if MODEL else 0
    })

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    API endpoint to get program recommendations
    
    Request body:
    {
        "fsc": 75.5,
        "matric": 80.0,
        "k": 4 (optional),
        "w_matric": 0.3 (optional),
        "w_fsc": 0.7 (optional)
    }
    
    Response:
    {
        "status": "success",
        "user_input": {...},
        "recommendations": [...],
        "summary": {...}
    }
    """
    try:
        # Validate model is loaded
        if MODEL is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded. Please train the model first.'
            }), 500
        
        # Parse request body
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        # Extract parameters
        fsc_score = data.get('fsc')
        matric_score = data.get('matric')
        k = data.get('k', 4)
        w_matric = data.get('w_matric', 0.3)
        w_fsc = data.get('w_fsc', 0.7)
        
        # Validate required fields
        if fsc_score is None or matric_score is None:
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields. Please provide both "fsc" and "matric"'
            }), 400
        
        # Validate score ranges
        if not (0 <= fsc_score <= 100) or not (0 <= matric_score <= 100):
            return jsonify({
                'status': 'error',
                'message': 'Scores must be between 0 and 100'
            }), 400
        
        # Validate k
        if not (1 <= k <= 20):
            return jsonify({
                'status': 'error',
                'message': 'k must be between 1 and 20'
            }), 400
        
        # Get recommendations
        recommendations = weighted_knn(
            matric_score=matric_score,
            fsc_score=fsc_score,
            k=k,
            w_matric=w_matric,
            w_fsc=w_fsc
        )
        
        # Calculate overall statistics
        user_avg_aggregate = np.mean([r['your_estimated_aggregate'] for r in recommendations])
        
        # Prepare response
        response = {
            'status': 'success',
            'user_input': {
                'matric_percentage': matric_score,
                'fsc_percentage': fsc_score,
                'weights': {
                    'matric': w_matric,
                    'fsc': w_fsc
                }
            },
            'recommendations': recommendations,
            'summary': {
                'total_recommendations': len(recommendations),
                'your_avg_aggregate': round(user_avg_aggregate, 2),
                'closest_match': recommendations[0]['program'] if recommendations else None,
                'closest_match_merit': recommendations[0]['closing_merit'] if recommendations else None,
                'best_chance': min([r['gap_to_closing'] for r in recommendations]) if recommendations else None
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/batch_recommend', methods=['POST'])
def batch_recommend():
    """
    Batch recommendation endpoint for multiple students
    
    Request body:
    {
        "students": [
            {"fsc": 75, "matric": 80},
            {"fsc": 85, "matric": 82}
        ],
        "k": 4
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'students' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Please provide "students" array in request body'
            }), 400
        
        students = data['students']
        k = data.get('k', 4)
        
        results = []
        for student in students:
            try:
                recommendations = weighted_knn(
                    matric_score=student['matric'],
                    fsc_score=student['fsc'],
                    k=k
                )
                results.append({
                    'student': student,
                    'recommendations': recommendations[:3]  # Top 3 for batch
                })
            except Exception as e:
                results.append({
                    'student': student,
                    'error': str(e)
                })
        
        return jsonify({
            'status': 'success',
            'batch_results': results
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/programs/list', methods=['GET'])
def list_programs():
    """List all available programs"""
    if MODEL is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    
    programs = [
        {
            'program': p['CAMPUS|PROGRAM'],
            'closing_merit': p['AGGR']
        }
        for p in MODEL['programs']
    ]
    
    return jsonify({
        'status': 'success',
        'total_programs': len(programs),
        'programs': programs
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get model statistics"""
    if MODEL is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    
    programs = MODEL['programs']
    aggr_values = [p['AGGR'] for p in programs]
    
    return jsonify({
        'status': 'success',
        'model_info': MODEL['metadata'],
        'statistics': {
            'total_programs': len(programs),
            'min_closing_merit': round(min(aggr_values), 2),
            'max_closing_merit': round(max(aggr_values), 2),
            'avg_closing_merit': round(np.mean(aggr_values), 2),
            'std_closing_merit': round(np.std(aggr_values), 2)
        }
    })

if __name__ == '__main__':
    # Load the model
    print("="*60)
    print("🚀 STARTING FLASK RECOMMENDATION SERVER")
    print("="*60)
    
    if load_model():
        print("\n📡 Starting API server on http://localhost:5000")
        print("🔗 Available endpoints:")
        print("   POST /recommend - Get program recommendations")
        print("   POST /batch_recommend - Batch recommendations")
        print("   GET  /programs/list - List all programs")
        print("   GET  /stats - Model statistics")
        print("   GET  /health - Health check")
        print("\n💡 Example request:")
        print('   curl -X POST http://localhost:5000/recommend \\')
        print('        -H "Content-Type: application/json" \\')
        print('        -d \'{"fsc": 75, "matric": 80}\'')
        print("\n" + "="*60)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to load model. Please run train_model.py first!")