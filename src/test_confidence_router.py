# test_confidence_router.py
# Testing confidence_router.py with dummy values
# Author: Anas

from confidence_router import route_prediction

# Test 1: High confidence - Normal
result1 = route_prediction([0.92, 0.08])
print("Test 1:", result1)

# Test 2: High confidence - Pneumonia
result2 = route_prediction([0.05, 0.95])
print("Test 2:", result2)

# Test 3: Low confidence - should go to Review
result3 = route_prediction([0.55, 0.45])
print("Test 3:", result3)

# Test 4: Border case - just below threshold
result4 = route_prediction([0.84, 0.16])
print("Test 4:", result4)