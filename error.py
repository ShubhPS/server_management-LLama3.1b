def load_user_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

def extract_emails(data):
    emails = []
    for line in data:
        if '@' in line:
            emails.append(line.email())  # This line will cause an error
    return emails

# Main function
if __name__ == "__main__":
    users = load_user_data("users.txt")
    emails = extract_emails(users)
    print("Extracted emails:", emails)
