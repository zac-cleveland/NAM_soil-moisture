#!/usr/bin/env python
# coding: utf-8

"""
This code loops through a "root directory", finds any python (.py) or jupyter notebook (.ipynb) file, looks through the code to find a specified "old directory", and updates that line the include the "new directory".  For example, if you have a directory path to some data, data_dir = /path/to/my/data, but you have made changes to your directory structure and the data now exists in /path/to/my/<new_subdirectory>/data, the code will find /path/to/my/data, and update it to /path/to/my/<new_subdirector>/data, so long as the root_dir, old_dir, and new_dir are properly specified before running the code.
"""

# import functions needed
import os
import glob
import fileinput

# define function to check existence of directories
def check_directories(root_dir, old_dir, new_dir):
    dir_dict = {
        "root_dir": os.path.esists(root_dir),
        "old_dir": os.path.exists(old_dir),
        "new_dir": os.path.exists(new_dir)
    }
    missing_dirs = [key for key, value in dir_dict.items() if not value]
    if missing_dirs:
        print("The following directories do not exist:")
        for dir in missing_dirs:
            print(dir)
        # verify with user to continue anyway
        cont = input('Continue anyway?'
                     '\n"y" for yes. Press any other key to abort:')
        if cont.lower() != 'y':
            print(f'You selected: {cont}. Code will not run.')
        else:
            return 

def verify_user


# define the function to find the .py and .ipynb files
def find_python_scripts(root_dir=None, old_dir=None, new_dir=None):
    # make sure all directories are specified and not left default
    if ((root_dir is None) or (old_dir is None) or (new_dir is None)):
        print(f'You must specify root_dir, old_dir, AND new_dir',
              f'\nroot_dir <{root_dir}>',
              f'\nold_dir <{old_dir}>',
              f'\nnew_dir <{new_dir}>')
        return

    # verify with user that the changes are correct to push through
    print(f'root_dir <{root_dir}>',
          f'\nold_dir <{old_dir}>',
          f'\nnew_dir <{new_dir}>\n')
    verify = input('Are you sure you want to continue with these updates?'
                   '\n"y" for yes. Press any other key to abort:')
    if verify.lower() != 'y':
        print(f'You selected: {cont}. Code will not run.')
        return

    # check existence of directories before proceeding
    dir_list = [root_dir, old_dir, new_dir]
    # create list of dirs that DO NOT exist
    missing_dirs = [dir_list[i] for i in range(len(dir_list)) if not os.path.exists(dir_list[i])]
    if any(missing_dirs):  # if any missing directories: user must re-verify to continue
        print('The following directories do not exist:')
        for i in range(len(missing_dirs)):
            print(missing_dirs[i])
        cont = input('Continue anyway?'
                     '\n"y" for yes. Press any other key to abort:')
        if cont.lower() != 'y':
            print(f'You selected: {cont}. Code will not run.')
            return

    if verify.lower() == 'y':
        # traverse through all directories and subdirectories
        for dir_path, _, files in os.walk(root_dir):
            for file_name in files:
                if file_name.endswith(('.py', '.ipynb')):
                    file_path = os.path.join(dir_path, file_name)
                    update_directory_in_script(file_path, old_dir, new_dir)
    else:
        print(f'You selected: {verify}.  Code will not run.')
        return

# define the function to replace old_dir with new_dir
def update_directory_in_script(file_path, old_dir, new_dir):
    # read in file contents
    with open(file_path, 'r') as file:
        file_contents = file.read()

    # replace old_dir with new_dir
    new_contents = file_contents.replace(old_dir, new_dir)

    # write updated contents back to file
    with open(file_path, 'w') as file:
        file.write(new_contents)
        
# run the code -- MAKE SURE TO DOUBLE CHECK SPECIFIED DIRECTORIES --
if __name__ == "__main__":
    # Specify your root directory and old/new directory paths
    root_directory = "/glade/u/home/zcleveland/NAM_soil-moisture"
    old_directory = "/EXAMPLE/PATH/FOR/DIR/UPDATE/FUNCTION/"
    new_directory = "/EXAMPLE/PATH/FOR/Hello_World/DIR/UPDATE/FUNCTION/"

    # Call the function to update directories in files
    find_python_scripts(root_directory, old_directory, new_directory)