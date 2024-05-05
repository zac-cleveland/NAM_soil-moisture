"""
This code loops through a "root directory", finds any python (.py) or jupyter notebook (.ipynb) file, looks through the code to find a specified string, "target_str", and if it exists, appends the filepath to a list.  
"""

import os
import fnmatch

def find_str_in_files(root_dir, target_str):
    # Initialize an empty list to store file paths
    file_list = []

    # Loop through the root directory and all subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # Check if the file is either a .py or .ipynb file
            if fnmatch.fnmatch(filename, '*.py') or fnmatch.fnmatch(filename, '*.ipynb'):
                filepath = os.path.join(dirpath, filename)

                # Check if the target string exists in the file
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                    if target_str in file.read():
                        # Append the full file path to the list
                        file_list.append(filepath)

    return file_list

if __name__ == "__main__":
    # Specify the root directory
    root_dir = "/path/to/your/root/directory"

    # Specify the target string
    target_str = "tcw_max_stats"

    # Find files containing the target string
    files_with_target = find_str_in_files(root_dir, target_str)

    # Print the list of file paths
    for filepath in files_with_target:
        print(filepath)





