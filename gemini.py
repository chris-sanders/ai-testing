#!/usr/bin/env python3
import google.generativeai as genai
import os
import glob
import mimetypes
import time
import argparse
import tempfile
from google.api_core import exceptions as google_exceptions

SUPPORTED_MIME_TYPES = [
    'text/plain', 
]

def get_mime_type(filepath):
    mime_type, _ = mimetypes.guess_type(filepath)
    if mime_type in SUPPORTED_MIME_TYPES:
        return mime_type
    return 'text/plain'  # Default to text/plain for unsupported types

def get_last_lines(filepath, max_lines=1000):
    """Get the last 'max_lines' lines from a file."""
    with open(filepath, 'rb') as file:
        return tail(file, max_lines)

def tail(file, lines):
    """Read the last 'lines' lines of a file efficiently."""
    BLOCK_SIZE = 1024
    file.seek(0, 2)
    block_end_byte = file.tell()
    lines_to_go = lines
    block_number = -1
    blocks = []
    while lines_to_go > 0 and block_end_byte > 0:
        if (block_end_byte - BLOCK_SIZE > 0):
            file.seek(block_number * BLOCK_SIZE, 2)
            blocks.append(file.read(BLOCK_SIZE))
        else:
            file.seek(0,0)
            blocks.append(file.read(block_end_byte))
        lines_found = blocks[-1].count(b'\n')
        lines_to_go -= lines_found
        block_end_byte -= BLOCK_SIZE
        block_number -= 1
    all_read_text = b''.join(reversed(blocks))
    return b'\n'.join(all_read_text.splitlines()[-lines:]).decode('utf-8', errors='replace')

def truncate_file(filepath, max_lines=1000):
    """Create a temporary file with the last 'max_lines' lines of the original file."""
    last_lines = get_last_lines(filepath, max_lines)
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
        temp_file.write(last_lines)
        return temp_file.name

def upload_file_with_retry(filepath, display_name, mime_type, max_retries=5, base_wait_time=5, max_lines=100, force_mime_type=None):
    truncated_filepath = truncate_file(filepath, max_lines)
    file_size = os.path.getsize(truncated_filepath)
    
    for attempt in range(max_retries):
        try:
            print(f"Attempting to upload last {max_lines} lines of {filepath} (Size: {file_size/1024/1024:.2f}MB)")
            start_time = time.time()
            file_response = genai.upload_file(path=truncated_filepath, display_name=display_name, mime_type=force_mime_type or mime_type)
            upload_time = time.time() - start_time
            print(f"Successfully uploaded truncated {filepath} in {upload_time:.2f} seconds")
            os.unlink(truncated_filepath)  # Delete the temporary file
            return file_response
        except Exception as e:
            print(f"Error uploading truncated {filepath}: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = base_wait_time * (2 ** attempt)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                os.unlink(truncated_filepath)  # Delete the temporary file
                raise

def count_tokens_with_retry(model, file_response, max_retries=5, base_wait_time=5):
    for attempt in range(max_retries):
        try:
            print(f"Counting tokens for {file_response.display_name}")
            token_count = model.count_tokens([file_response])
            print(f"Token count for {file_response.display_name}: {token_count.total_tokens}")
            return token_count.total_tokens
        except google_exceptions.DeadlineExceeded:
            if attempt < max_retries - 1:
                wait_time = base_wait_time * (2 ** attempt)
                print(f"Token counting timed out. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to count tokens after {max_retries} attempts")
                raise
        except Exception as e:
            print(f"Error counting tokens: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = base_wait_time * (2 ** attempt)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise

def upload_files(directory, model, force_mime_type=None):
    uploaded_files = []
    skipped_files = []
    total_tokens = 0
    max_tokens = 2000000  # 2M token limit

    for filepath in glob.glob(f"{directory}/**/*", recursive=True):
        if os.path.isfile(filepath):
            relative_path = os.path.relpath(filepath, directory)
            mime_type = get_mime_type(filepath)
            print(f"File: {relative_path}, MIME type: {mime_type}")
            try:
                file_response = upload_file_with_retry(filepath, relative_path, mime_type, force_mime_type=force_mime_type)
                
                try:
                    file_tokens = count_tokens_with_retry(model, file_response)
                    
                    if total_tokens + file_tokens > max_tokens:
                        print(f"Skipping {filepath}: Would exceed token limit")
                        skipped_files.append(filepath)
                        genai.delete_file(name=file_response.name)
                        continue

                    uploaded_files.append(file_response)
                    total_tokens += file_tokens
                    print(f"Processed file {file_response.display_name} (MIME: {mime_type}, Tokens: {file_tokens})")
                    print(f"Total tokens used: {total_tokens}/{max_tokens}")
                except Exception as e:
                    print(f"Failed to process {filepath}: {str(e)}")
                    genai.delete_file(name=file_response.name)
                    skipped_files.append(filepath)
            except Exception as e:
                print(f"Failed to upload {filepath}: {str(e)}")
                skipped_files.append(filepath)

    return uploaded_files, skipped_files, total_tokens

def chat_with_model(model, system_prompt, file_objects, max_retries=3):
    chat = model.start_chat(history=[])
    
    print("Chat started. Type 'exit' or press Ctrl+D to end the conversation.")
    print("AI: Hello! I'm ready to help you with any questions about the uploaded files. What would you like to know?")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            
            full_prompt = [
                system_prompt,
                f"User question: {user_input}",
                "Please analyze the attached files to answer the question."
            ]

            for attempt in range(max_retries):
                try:
                    print(f"Sending request (attempt {attempt + 1}/{max_retries})...")
                    response = chat.send_message(full_prompt + file_objects)
                    print("AI:", response.text)
                    break
                except google_exceptions.DeadlineExceeded:
                    if attempt < max_retries - 1:
                        print(f"Request timed out. Retrying in {5 * (attempt + 1)} seconds...")
                        time.sleep(5 * (attempt + 1))
                    else:
                        print("Maximum retries reached. The model is taking too long to respond.")
                        print("Consider simplifying your query or reducing the number of files.")
                except Exception as e:
                    print(f"An error occurred: {str(e)}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {5 * (attempt + 1)} seconds...")
                        time.sleep(5 * (attempt + 1))
                    else:
                        print("Maximum retries reached. Unable to get a response.")
                        break

        except EOFError:
            print("\nExiting chat...")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            print("Continuing chat...")

def cleanup_files(uploaded_files):
    for file in uploaded_files:
        try:
            genai.delete_file(name=file.name)
            print(f'Deleted file {file.display_name}')
        except Exception as e:
            print(f"Failed to delete {file.display_name}: {str(e)}")

def list_uploaded_files():
    try:
        file_list = genai.list_files()
        return file_list
    except Exception as e:
        print(f"An error occurred while listing files: {e}")
        return []

def delete_file(file_id):
    try:
        genai.delete_file(file_id)
        print(f"File with ID {file_id} has been successfully deleted.")
    except Exception as e:
        print(f"An error occurred while deleting the file: {e}")

def cleanup_all_files():
    files = list_uploaded_files()
    for file in files:
        delete_file(file.name)

def verify_mime_types(directory):
    for filepath in glob.glob(f"{directory}/**/*", recursive=True):
        if os.path.isfile(filepath):
            mime_type = get_mime_type(filepath)
            relative_path = os.path.relpath(filepath, directory)
            print(f"File: {relative_path}")
            print(f"  MIME type: {mime_type}")
            if mime_type not in SUPPORTED_MIME_TYPES:
                print("  WARNING: Unsupported MIME type, will be treated as text/plain")

def main():
    parser = argparse.ArgumentParser(description="Upload files and chat with Gemini model.")
    parser.add_argument("directory", help="Directory containing files to upload")
    parser.add_argument("--verify-mime", action="store_true", help="Verify MIME types before uploading")
    parser.add_argument("--force-mime", help="Force all files to use this MIME type")
    args = parser.parse_args()

    # Get API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    # Initialize Google API Client
    genai.configure(api_key=api_key)
    model_name = "models/gemini-1.5-pro-latest"
    model = genai.GenerativeModel(model_name=model_name)

    if args.verify_mime:
        print("Verifying MIME types...")
        verify_mime_types(args.directory)
        proceed = input("Do you want to proceed with the upload? (y/n): ")
        if proceed.lower() != 'y':
            print("Aborting upload.")
            return

    cleanup_all_files()
    uploaded_files, skipped_files, total_tokens = upload_files(args.directory, model, force_mime_type=args.force_mime)

    if total_tokens >= 2000000:
        print("Warning: Maximum token limit reached. Some files may have been skipped.")

    if skipped_files:
        print("The following files were skipped:")
        for file in skipped_files:
            print(f"- {file}")

    system_prompt = ("You are an AI assistant with access to several uploaded files. "
                     "When the user asks a question, you can analyze and refer to these files to provide information. "
                     "The files may contain logs, configuration data, or other relevant information.")

    chat_with_model(model, system_prompt, uploaded_files)
    cleanup_files(uploaded_files)

if __name__ == "__main__":
    main()
