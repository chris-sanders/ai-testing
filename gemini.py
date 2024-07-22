#!/usr/bin/env python3

import google.generativeai as genai
import os
import argparse
import glob
import mimetypes

# Get API key from environment variable
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

# Initialize Google API Client
genai.configure(api_key=api_key)

# List of supported MIME types
SUPPORTED_MIME_TYPES = [
    'text/plain', 'text/html', 'text/css', 'text/javascript',
    'application/x-javascript', 'text/x-typescript', 'application/x-typescript',
    'text/csv', 'text/markdown', 'text/x-python', 'application/x-python-code',
    'application/json', 'text/xml', 'application/rtf', 'text/rtf'
]

def get_mime_type(filepath):
    mime_type, _ = mimetypes.guess_type(filepath)
    if mime_type in SUPPORTED_MIME_TYPES:
        return 'text/plain'  # Treat YAML as plain text
    else:
        return 'text/plain'  # Default to plain text for all other files

def upload_files(directory):
    uploaded_files = []
    skipped_files = []
    for filepath in glob.glob(f"{directory}/**/*", recursive=True):
        if os.path.isfile(filepath):
            relative_path = os.path.relpath(filepath, directory)
            display_name = relative_path
            mime_type = get_mime_type(filepath)
            try:
                file_response = genai.upload_file(path=filepath, display_name=display_name, mime_type=mime_type)
                uploaded_files.append((file_response, mime_type))
                print(f"Uploaded file {file_response.display_name} as: {file_response.uri} (MIME: {mime_type})")
            except Exception as e:
                print(f"Failed to upload {filepath}: {str(e)}")
                skipped_files.append((filepath, mime_type))
    return uploaded_files, skipped_files

def chat_with_model(model, system_prompt, file_objects):
    chat = model.start_chat(history=[])
    
    print("Chat started. Type 'exit' or press Ctrl+D to end the conversation.")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            
            # Combine system prompt, user input, and files for each message
            full_prompt = [
                system_prompt,
                f"User question: {user_input}",
                "Use only information from the attached files and user input to answer the question."
            ]
            response = chat.send_message(full_prompt + file_objects)
            print("AI:", response.text)
        except EOFError:
            print("\nExiting chat...")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Continuing chat...")

def cleanup_files(uploaded_files):
    for file, _ in uploaded_files:
        try:
            genai.delete_file(name=file.name)
            print(f'Deleted file {file.display_name}')
        except Exception as e:
            print(f"Failed to delete {file.display_name}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Upload files and chat with Gemini model.")
    parser.add_argument("directory", help="Directory containing files to upload")
    args = parser.parse_args()

    print(f"Args initialized: {args}")
    uploaded_files, skipped_files = upload_files(args.directory)

    # Extract just the file objects from the list of tuples
    file_objects = [file for file, _ in uploaded_files]

    model_name = "models/gemini-1.5-pro-latest"
    model = genai.GenerativeModel(model_name=model_name)

    # Create a system prompt
    system_prompt = ("You are a Kubernetes expert helping people identify root causes for issues in logs. Always provide reasoning for your answers and ask clarifying questions if needed.")

    chat_with_model(model, system_prompt, file_objects)
    cleanup_files(uploaded_files)

if __name__ == "__main__":
    main()
