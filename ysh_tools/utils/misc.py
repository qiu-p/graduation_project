'''
time 
file & dir
'''

# import numpy as np
import os
import shutil
import random
import time
from os import path as osp


def get_time_str() -> str:
    '''
    %Y%m%d_%H%M%S
    '''
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

def split_path(path):
    '''
    eg. 'AAA/BBB/CCC.txt' -> 'AAA/BBB', 'CCC', '.txt'
    '''
    pre_path = os.path.dirname(path)
    name_without_ext, ext = os.path.splitext(os.path.basename(path))

    return pre_path, name_without_ext, ext

#############################

def rename_dir_or_file(path, template='{}_archived_#t'):
    '''
    - Description:
      - Rename the name of the path.
    - Args:
      - path (str): dir path
      - template (str): the template to rename the path
      - delete (bool): if path exists, whether to delete it
    - Return:
      - (bool): whether the path exists
    - Details:
      - If you want to express time, you can use '#t'.
      - If you just want to express '#", you can use '##'.
    '''
    if osp.exists(path):
        detect_pound = False
        i = 0
        for char in template:
            if detect_pound == True:
                if char == 't':
                    time_str = get_time_str()
                    if len(template) == i+1:
                        template = template[:i-1] + time_str
                    else:
                        template = template[:i-1] + time_str + template[i+1:]
                    i = i + len(time_str) - 2
                elif char == '#':
                    template = template[:i-1] + template[i:]
                    i = i - 1
                detect_pound = False
            elif char == '#':
                detect_pound = True
            i = i + 1
        
        pre_path, name_without_ext, ext = split_path(path)
        new_name = pre_path + '/' + template.format(name_without_ext) + ext
        os.rename(path, new_name)
        return True
    else:
        return False

def add_suffix(path, suffix):
    '''
    - Description:
      - add suffix to dir name or file name
    - Args:
      - path (str): eg. AAA/BBB/CC.txt
      - index (str): eg. 1
    - Return:
        eg. AAA/BBB/CC_1.txt
    '''
    filename_tmpl = '{}_' + suffix
    rename_dir_or_file(path, filename_tmpl)

def make_dir(dir_path, mode='e', template='{}_archived_#t'):
    """
    - Description:
      - mkdirs
    - Args:
      - path (str): Folder path.
      - mode (str): 'e' 'i' 'r' or 'd'.
        - e(default): throw ERROR if dir has existed
        - i: (ignore) if dir has existed, just return and do nothing
        - r: rename the old dir if dir has existed
        - d: delete the old dir if dir has existed 
      - template (str): the template to rename the path
        - If you want to express time, you can use '#t'.
        - If you just want to express '#", you can use '##'. 
        - default: '{}_archived_#t'
    - Return:
      - is_exist (bool): whether the dir has existed
    """
    is_exist = False
    if osp.exists(dir_path):
        is_exist = True
        if mode == 'r':
            rename_dir_or_file(dir_path, template)
        elif mode == 'd':
            shutil.rmtree(dir_path)
        elif mode == 'e':
            raise FileExistsError('the dir has existed')
        elif mode == 'i':
            return is_exist
        else:
            raise ValueError('\'{}\' is not a mode. Mode selection: e, r, d.'.format(mode))
    # exist_ok：是否在目录存在时触发异常。
    # exist_ok = False（默认值），则在目标目录已存在的情况下触发 FileExistsError 异常；
    # exist_ok = True，则在目标目录已存在的情况下不会触发 FileExistsError 异常。
    os.makedirs(dir_path)
    return is_exist

def scan_dir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scan_dir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scan_dir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scan_dir(dir_path, suffix=suffix, recursive=recursive)

##############################################

def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def sizeof_fmt(size, suffix='B'):
    """Get human readable file size.

    Args:
        size (int): File size.
        suffix (str): Suffix. Default: 'B'.

    Return:
        str: Formatted file size.
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(size) < 1024.0:
            return f'{size:3.1f} {unit}{suffix}'
        size /= 1024.0
    return f'{size:3.1f} Y{suffix}'

def get_from_dict(dict, key, replace_none = None):
    """ 
    get value from dictionary
    Aegs:
        dict (dict):
        key (int, str, ...):
        replace_none: the value to replace the return if the original value is None
    Return:
        (bool): whethe the key exists
        (value): value 
    """
    if key not in dict:
        return False, None
    else:
        value = dict.get(key)
        if value == None:
            value = replace_none
        return True, value
