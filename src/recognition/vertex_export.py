import json
import os
import argparse
import glob


def convert_to_vertex_jsonl(input_dir, gcs_image_base_uri, output_file):
    """
    Converts a directory of individual COCO-like JSON annotation files
    into a single Vertex AI-compatible JSON Lines (.jsonl) file.

    Each JSON file in the input directory is assumed to represent one image.
    This version de-duplicates annotations and ONLY includes 'visible: true' annotations.

    Args:
        input_dir (str): Path to the directory containing the 5000+ JSON files.
        gcs_image_base_uri (str): The GCS URI prefix where the corresponding
                                  images are stored.
                                  (e.g., "gs://my-bucket/my-images/")
        output_file (str): The name of the output .jsonl file to be created.
    """

    # Ensure the GCS URI has a trailing slash
    if not gcs_image_base_uri.endswith('/'):
        gcs_image_base_uri += '/'

    # Find all JSON files in the input directory
    # Use glob to handle large numbers of files efficiently
    json_files = glob.glob(os.path.join(input_dir, '*.json'))

    if not json_files:
        print(f"Error: No .json files found in directory: {input_dir}")
        return

    print(f"Found {len(json_files)} JSON files. Starting conversion...")

    processed_count = 0
    # Open the output .jsonl file in write mode
    with open(output_file, 'w') as f_out:
        for json_path in json_files:
            try:
                # Read and parse the individual JSON file
                with open(json_path, 'r') as f_in:
                    data = json.load(f_in)

                # --- 1. Construct Image GCS URI ---
                image_filename = data['image']['file_name']
                image_gcs_uri = f"{gcs_image_base_uri}{image_filename}"

                # --- 2. Process Annotations (with De-duplication and Visibility Check) ---
                vertex_annotations = []
                seen_annotations = set()  # To track duplicates

                for ann in data['annotations']:

                    # --- ADDED: Check for visibility ---
                    # Per your request, only add annotations where "visible" is True.
                    if not ann.get('visible', False):  # .get() safely handles missing keys
                        continue  # Skip this annotation if it's not visible
                    # --- End of added code ---

                    # Use the normalized bounding box
                    bbox_norm = ann['bbox_normalized']

                    # Convert [x_min, y_min, width, height] to [x_min, y_min, x_max, y_max]
                    x_min = bbox_norm[0]
                    y_min = bbox_norm[1]
                    width = bbox_norm[2]
                    height = bbox_norm[3]

                    # Calculate x_max and y_max, ensuring they don't exceed 1.0
                    x_max = min(x_min + width, 1.0)
                    y_max = min(y_min + height, 1.0)

                    # Create a unique tuple for this annotation
                    # Round to 6 decimal places to avoid floating point inaccuracies
                    ann_tuple = (
                        ann['category_name'],
                        round(x_min, 6),
                        round(y_min, 6),
                        round(x_max, 6),
                        round(y_max, 6)
                    )

                    # Check if we've already added this exact annotation
                    if ann_tuple not in seen_annotations:
                        # Create the annotation format required by Vertex AI
                        vertex_ann = {
                            "display_name": ann['category_name'],
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max
                        }
                        vertex_annotations.append(vertex_ann)
                        seen_annotations.add(ann_tuple)  # Add to our set of seen annotations
                    else:
                        # Silently skip the duplicate
                        pass

                        # --- 3. Create the JSON Lines object for this image ---

                # --- ADDED: Check if any annotations were actually added ---
                if not vertex_annotations:
                    # This image had no 'visible' annotations, so we skip it.
                    print(f"Info: No visible annotations found for {image_filename}. Skipping this file.")
                    continue  # Go to the next json_path
                # --- End of added code ---

                jsonl_item = {
                    "image_gcs_uri": image_gcs_uri,
                    "bounding_box_annotations": vertex_annotations
                }

                # --- 4. Write the object as a JSON string to the output file ---
                # Each line in the .jsonl file must be a complete JSON object
                f_out.write(json.dumps(jsonl_item) + '\n')
                processed_count += 1

            except Exception as e:
                print(f"Warning: Could not process file {json_path}. Error: {e}")

    print(f"\nConversion complete!")
    print(f"Output file created: {output_file}")
    print(f"Total images processed: {processed_count}")


if __name__ == "__main__":

    # Run the conversion function
    convert_to_vertex_jsonl(r"C:\Users\heszl\repos\travel-squares\src\training_data_generator\training_data\annotations", "gs://square-pirates/images/images/", r"C:\Users\heszl\repos\travel-squares\src\training_data_generator\training_data\square-pirates.jsonl")

    print("\n--- How to use this script ---")
    print("1. Make sure all your 5000+ .json files are in one directory.")
    print("2. Upload all your 5000+ .png image files to a GCS bucket.")
    print("3. Run this script from your terminal, providing the paths:")
    print(
        f"   python {os.path.basename(__file__)} --input_dir /path/to/your/jsons --gcs_uri gs://your-bucket/path/to/images/")
    print("\nThis will create the 'vertex_ai_annotations.jsonl' file in the same directory.")