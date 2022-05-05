import os, sys



def load_from_dotenv(variable_name):
    ''' The dotenv module is incompatible with some PyVertical dependencies. This function can be used to load environment variables instead. It throws an exception if 
    '''
    path = os.path.abspath(os.path.dirname(sys.argv[0]))
    if path[-1] != '/': path += '/'
    dotenv_file = path + '.env'
    with open(dotenv_file,'r') as f:
        for line in f:
            if variable_name in line:
                try: 
                    try:
                        return line.split('"')[1]
                    except:
                        return line.split("'")[1]
                except IndexError:
                    raise EnvironmentVariableNotSetError(f'The environment variable {variable_name} is not available in the file {dotenv_file}')


