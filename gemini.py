#!/usr/bin/env python3

import google.generativeai as genai
import os
import argparse
import glob

# Get API key from environment variable
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

# Initialize Google API Client
genai.configure(api_key=api_key)

def upload_files(directory):
    uploaded_files = []
    for filepath in glob.glob(f"{directory}/**/*", recursive=True):
        if os.path.isfile(filepath):
            relative_path = os.path.relpath(filepath, directory)
            display_name = relative_path
            file_response = genai.upload_file(path=filepath, display_name=display_name)
            uploaded_files.append(file_response)
            print(f"Uploaded file {file_response.display_name} as: {file_response.uri}")
    return uploaded_files

#def chat_with_model(model, uploaded_files):
#    chat = model.start_chat(history=[])
#    
#    print("Chat started. Type 'exit' or press Ctrl+D to end the conversation.")
#    while True:
#        try:
#            user_input = input("You: ")
#            if user_input.lower() == 'exit':
#                break
#            
#            response = chat.send_message([user_input] + uploaded_files)
#            print("AI:", response.text)
#        except EOFError:
#            # This catches the Ctrl+D input
#            print("\nExiting chat...")
#            break

def cleanup_files(uploaded_files):
    for file in uploaded_files:
        genai.delete_file(name=file.name)
        print(f'Deleted file {file.display_name}')

def chat_with_model(model, uploaded_files):
    chat = model.start_chat(history=[])
    
    print("Chat started. Type 'exit' or press Ctrl+D to end the conversation.")
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            
            # Only send text input to the model, not the uploaded files
            response = chat.send_message(user_input)
            print("AI:", response.text)
        except EOFError:
            print("\nExiting chat...")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Continuing chat...")

def main():
    parser = argparse.ArgumentParser(description="Upload files and chat with Gemini model.")
    parser.add_argument("directory", help="Directory containing files to upload")
    args = parser.parse_args()

    uploaded_files = upload_files(args.directory)

    model_name = "models/gemini-1.5-pro-latest"
    model = genai.GenerativeModel(model_name=model_name)

    # Initialize the chat with the uploaded files
    try:
        chat = model.start_chat(history=[])
        context = "I have uploaded some files. Please use them as context for our conversation."
        response = chat.send_message([context] + uploaded_files)
        print("AI:", response.text)
    except Exception as e:
        print(f"Error initializing chat with files: {str(e)}")
        print("Continuing without file context...")

    chat_with_model(model, [])  # Pass an empty list instead of uploaded_files

    cleanup_files(uploaded_files)

if __name__ == "__main__":
    main()
