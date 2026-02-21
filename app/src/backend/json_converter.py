import json
import os

def export_segments_to_txt(json_input_path, output_dir):
    """
    Reads a JSON file of transcript segments and saves the text 
    to a .txt file in the specified output directory.
    """
    # 1. Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load the JSON data
    try:
        with open(json_input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    # 3. Extract the text from each segment
    # Joining with a newline so each segment is on its own line
    full_text = "\n".join([segment.get("text", "") for segment in data])

    # 4. Prepare the output filename
    # This takes 'filename.json' and turns it into 'filename.txt'
    base_name = os.path.splitext(os.path.basename(json_input_path))[0]
    output_filename = f"{base_name}.txt"
    output_path = os.path.join(output_dir, output_filename)

    # 5. Write to the .txt file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"Success! Text exported to: {output_path}")
    except Exception as e:
        print(f"Error writing TXT file: {e}")

if __name__ == "__main__":
    # Example Usage:
    # Use the 'r' prefix for Raw strings
    input_file = r"C:\files\podcast\app\files\test\Walid - Outdoor Interview - UGC - UnlimitEd\Walid - Outdoor Interview - UGC - UnlimitEd_full.json"
    target_folder = r"C:\files\podcast\app\files\test\output_texts"
    
    export_segments_to_txt(input_file, target_folder)