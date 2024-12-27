import os

print(os.listdir('.'))  # Check if 'model.h5' is listed exactly
print(os.getcwd())  # Prints the directory where the script is running
from tensorflow.keras.models import load_model
loaded_model = load_model('D:\study\codes\engineering_codes\projects\written_digit_recognition_using_c\codes\model.h5')
