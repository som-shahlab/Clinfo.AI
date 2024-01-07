import json
from werkzeug.security import generate_password_hash

def save_json(dict_:dict,file_name:str):
    with open(file_name, 'w') as fp:
        json.dump(dict_, fp)
        
        
def read_json(file_name:str):
    with open(file_name, 'r') as fp:
        data = json.load(fp)
    return data


def retirive_user(user_emails,token="_1265!"):
    users = {}
    for email in user_emails:
        password = email.split('@')[0] + token
        users[email] = {"password": generate_password_hash(password) }

    return users