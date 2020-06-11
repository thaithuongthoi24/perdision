import joblib
import sys
regressor = joblib.load('./py-code/price_predict_model_file.pkl')

# props = [2000,900,900,9000,2005,2,60,8,4]
# props = [38,4,4,1,2,2,2,2,3,20]
props = [sys.argv[1],sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9],sys.argv[10]]
value = regressor.predict([props])

print(value)


