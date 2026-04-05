import jwt
import time

# Create a valid local JWT token for Danu based on standard Moodle plugin payloads
token = jwt.encode(
    {
        'user_id': '6', # Typical ID for secondary user
        'username': 'danu', 
        'role': 'moodle_user', 
        'exp': time.time() + 3600
    }, 
    'your-super-secret-jwt-key-for-local-dev', 
    algorithm='HS256'
)
print(token)
