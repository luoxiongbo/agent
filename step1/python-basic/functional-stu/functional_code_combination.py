def normalize(str):
    return str.strip().lower()

def is_valid_email(str):
    return '@' in str and '.' in str

emails = ['a@develop.com', 'bad', 'b@test.com']

valid_emails = [
    normalize(e)
    for e in emails
    if is_valid_email(normalize(e))
]

print(valid_emails)