import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Testing Multi-Model Support...")
try:
    from model import classifier, get_models, switch_model, get_active_model
    
    # Test 1: Check initial model
    print("\n=== Test 1: Initial Model ===")
    active = get_active_model()
    print(f"Active model: {active}")
    
    # Test 2: Get available models
    print("\n=== Test 2: Available Models ===")
    models = get_models()
    for key, config in models.items():
        print(f"  - {key}: {config['description']}")
        print(f"    Type: {config['type']}")
    
    # Test 3: Check classifier state
    print("\n=== Test 3: Classifier State ===")
    if classifier.active_model is None:
        print("ERROR: Active model is None!")
        sys.exit(1)
    else:
        print(f"Classifier ready with model: {classifier.active_model_key}")
        print(f"Model classes: {classifier.active_model.names}")
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
