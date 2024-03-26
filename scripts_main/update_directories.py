"""
This code loops through a "root directory", finds any python (.py) or jupyter notebook (.ipynb) file, looks through the code to find a specified "old directory", and updates that line the include the "new directory".  For example, if you have a directory path to some data, data_dir = /path/to/my/data, but you have made changes to your directory structure and the data now exists in /path/to/my/<new_subdirectory>/data, the code will find /path/to/my/data, and update it to /path/to/my/<new_subdirector>/data, so long as the root_dir, old_dir, and new_dir are properly specified before running the code.
"""

# import needed functions
import sys
import os

sys.path.append('/glade/u/home/zcleveland/NAM_soil-moisture/scripts_main/')

# define root_directory, old_directory, and new_directory
root_dir = '/glade/u/home/zcleveland/NAM_soil-moisture/'
old_dir = None
new_dir = None


# define function to check existence of directories
# return list of missing directories
def check_directories(root_dir, old_dir, new_dir):
    dir_dict = {
        "root_dir": os.path.exists(root_dir),
        "old_dir": os.path.exists(old_dir),
        "new_dir": os.path.exists(new_dir)
    }
    missing_dirs = [key for key, value in dir_dict.items() if not value]
    if missing_dirs:
        print("The following directories do not exist:\n")
        for dir in missing_dirs:
            print(dir)
    else:
        print("All directories verified to exist:\n")
        print(f'\nroot_dir <{root_dir}>',
              f'\nold_dir <{old_dir}>',
              f'\nnew_dir <{new_dir}>')
    return missing_dirs


# define function to make user verify they want to continue
def verify_continue():
    user_input = input('Continue with changes? This is the LAST check.'
                       '\n"y" for yes. Press any other key to abort:')
    if user_input.lower() != 'y':  # abort code
        print(f'\nYou selected: "{user_input}". Code will not run.')
        return False
    else:  # continue code
        print(f'\nYou selected: "{user_input}". Code will continue.')
        return True


# define the function to find the .py and .ipynb files
def find_python_scripts(root_dir):
    py_file_paths = []
    # traverse through all directories and subdirectories
    for dir_path, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.endswith(('.py', '.ipynb')):
                py_file_paths.append(os.path.join(dir_path, file_name))
    # return python and jupyter notebook files
    return py_file_paths


# define a function to find which files will be updated
def files_to_update(py_file_paths, old_dir, new_dir):
    update_files = []
    # loop through py files and read in contents
    for file_path in py_file_paths:
        # read in contents
        with open(file_path, 'r') as file:
            file_contents = file.read()

        # replace old contents with new contents
        new_contents = file_contents.replace(old_dir, new_dir)
        if new_contents != file_contents:  # only append to list if file and new contents are different
            update_files.append(file_path)
    print('\nThe following files will be updated:\n')
    for file in update_files:
        print(file)

    return update_files  # only files that will be updated


# define a function to replace old_dir with new_dir
def update_py_files(update_files, old_dir, new_dir):
    # loop through py files and update them
    for py_file in update_files:
        # open files and read contents
        with open(py_file, 'r') as file:
            file_contents = file.read()

        new_contents = file_contents.replace(old_dir, new_dir)
        # write updated contents back to file
        with open(py_file, 'w') as file:
            file.write(new_contents)


# define the function to actually run all of this code
def update_directories(root_dir, old_dir, new_dir):

    # check for any non-existent directories and abort if any don't exist
    if check_directories(root_dir, old_dir, new_dir):
        check_override = input('\nWould you like to override and continue anyway?'
                               '\n"y" for yes. Press any other key to abort: ')
        if check_override.lower() != 'y':
            print('\nCode will not continue.')
            return
        else:
            print('\nOverride confirmed.  Code will continue.')

    # find all python and jupyter notebook files in root_dir
    py_file_paths = find_python_scripts(root_dir)

    # list files that will be updated
    update_files = files_to_update(py_file_paths, old_dir, new_dir)

    # verify the user wants to continue
    if not verify_continue():  # abort
        return
    else:  # continue with updates
        update_py_files(update_files, old_dir, new_dir)


# run the code
if __name__ == '__main__':
    update_directories(root_dir, old_dir, new_dir)


