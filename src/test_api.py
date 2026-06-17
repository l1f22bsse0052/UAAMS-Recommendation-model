import requests
import json

# API endpoint
BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())
    print()

def test_recommend():
    """Test single recommendation"""
    url = f"{BASE_URL}/recommend"
    
    # Test case 1: Average student
    payload1 = {
        "fsc": 75,
        "matric": 80,
        "k": 4
    }
    
    print("="*60)
    print("TEST CASE 1: Average Student")
    print("FSc: 75%, Matric: 80%")
    print("="*60)
    
    response = requests.post(url, json=payload1)
    result = response.json()
    
    if result['status'] == 'success':
        print(f"\n📊 Your estimated aggregate: {result['summary']['your_avg_aggregate']}%")
        print(f"\n🎯 Top {len(result['recommendations'])} Recommendations:\n")
        
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"{i}. {rec['program']}")
            print(f"   📍 Closing Merit: {rec['closing_merit']}%")
            print(f"   📈 Est. Matric needed: {rec['estimated_matric_needed']}%")
            print(f"   📉 Est. FSc needed: {rec['estimated_fsc_needed']}%")
            print(f"   🎯 Your aggregate: {rec['your_estimated_aggregate']}%")
            print(f"   📊 Gap: {rec['gap_to_closing']:+.2f}%")
            print(f"   🤝 Similarity: {rec['similarity_distance']}")
            print()
    else:
        print("Error:", result.get('message'))
    
    print("\n" + "="*60)
    print("TEST CASE 2: High Achiever")
    print("FSc: 90%, Matric: 88%")
    print("="*60)
    
    payload2 = {
        "fsc": 90,
        "matric": 88,
        "k": 4
    }
    
    response = requests.post(url, json=payload2)
    result = response.json()
    
    if result['status'] == 'success':
        print(f"\n📊 Your estimated aggregate: {result['summary']['your_avg_aggregate']}%")
        print(f"\n🎯 Top {len(result['recommendations'])} Recommendations:\n")
        
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"{i}. {rec['program']}")
            print(f"   📍 Closing Merit: {rec['closing_merit']}%")
            print(f"   🎯 Gap: {rec['gap_to_closing']:+.2f}%")
            print()

def test_batch():
    """Test batch recommendations"""
    url = f"{BASE_URL}/batch_recommend"
    
    payload = {
        "students": [
            {"fsc": 75, "matric": 80},
            {"fsc": 85, "matric": 82},
            {"fsc": 65, "matric": 70}
        ],
        "k": 3
    }
    
    print("\n" + "="*60)
    print("BATCH RECOMMENDATIONS")
    print("="*60)
    
    response = requests.post(url, json=payload)
    result = response.json()
    
    if result['status'] == 'success':
        for batch in result['batch_results']:
            if 'error' in batch:
                print(f"\nStudent {batch['student']}: Error - {batch['error']}")
            else:
                print(f"\nStudent (Matric={batch['student']['matric']}%, FSc={batch['student']['fsc']}%)")
                print("Top 3 Recommendations:")
                for i, rec in enumerate(batch['recommendations'], 1):
                    print(f"  {i}. {rec['program'][:50]} - {rec['closing_merit']}%")
    else:
        print("Error:", result.get('message'))

def test_stats():
    """Test statistics endpoint"""
    response = requests.get(f"{BASE_URL}/stats")
    print("\n" + "="*60)
    print("MODEL STATISTICS")
    print("="*60)
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    print("🚀 Testing Recommendation API")
    print("="*60)
    
    # Test health first
    test_health()
    
    # Test single recommendation
    test_recommend()
    
    # Test batch
    test_batch()
    
    # Test stats
    test_stats()
