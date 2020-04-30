from distutils.util import strtobool
import sys
import os
def user_yes_no_query(question):
    sys.stdout.write('%s [y/n]\n' % question)
    while True:
        try:
            return strtobool(input().lower())
        except ValueError:
            sys.stdout.write('Please respond with \'y\' or \'n\'.\n')

def check_directory(dir_name, dbg=False):
        try:
            # Create target Directory
            os.mkdir(dir_name)
            if dbg: print("Directory ", dir_name, " Created ")
        except FileExistsError:
            if dbg: print("Directory ", dir_name, " already exists")
        # end try
    # end function