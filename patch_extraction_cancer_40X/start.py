import os
import sys
import time
import random
import traceback

def touch_file(fn):
    os.system('touch ' + fn)

def rm_file(fn):
    os.system('rm -f ' + fn)

def rm_folder(fol):
    os.system('rm -rf ' + fol )

def create_fol(fol):
    if not os.path.exists(fol):
        os.mkdir(fol)

def clean_files(fol, limit_time = 60):
    files = {f for f in os.listdir(fol) if not f.startswith('.')}
    for fn in files:
        fn_path = os.path.join(fol, fn)
        (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(fn_path)
        if (time.time() - ctime)/60 > limit_time: # have been created more than 2hrs ago
            rm_file(os.path.join(fol, fn))

def list_files(fol):
    files = [f for f in os.listdir(fol) if not f.startswith('.')]
    return files

def is_path_exists(*args):
    path = '/'.join(args)
    return os.path.exists(path)

def clean_done_fol(done_fol, out_fol, indicator_file='extraction_done.txt'):
    done_fns = list_files(done_fol)
    out_fns = list_files(out_fol)
    for fn in done_fns:
        if not is_path_exists(out_fol, fn, indicator_file):
            rm_file(os.path.join(done_fol, fn))
    for fn in out_fns:
        if is_path_exists(out_fol, fn, indicator_file) and not is_path_exists(done_fol, fn):
            touch_file(os.path.join(done_fol, fn))

if __name__ == '__main__':
    IN_FOLDER = sys.argv[1]
    OUT_FOLDER = sys.argv[2]
    LOG_OUTPUT_FOLDER = sys.argv[3]
    indicator_file='extraction_done.txt'

    done_fol = 'done'
    processing_fol = 'processing'
    create_fol(done_fol)
    create_fol(processing_fol)
    clean_done_fol(done_fol, OUT_FOLDER, indicator_file)

    time.sleep(random.randint(100, 1000)/100.0)  # wait for 1 --> 10s to avoid concurrency
    start_time = time.time()
    while(1):
        elapsed_time = time.time() - start_time
        if elapsed_time > 100*60: # require at least 20mins left to start processing a new WSI
            exit(0)

        svs_done = set(list_files(done_fol))
        svs_processing = set(list_files(processing_fol))
        svs_all = set(list_files(IN_FOLDER))
        svs_remaining = svs_all.difference(svs_done.union(svs_processing))
        if len(svs_remaining) == 0:
            exit(0)

        clean_files(processing_fol, 60)

        svs_remaining = list(svs_remaining)
        random.shuffle(svs_remaining)
        slide_name = svs_remaining[0]
        log_path = os.path.join(LOG_OUTPUT_FOLDER, 'log.save_svs_to_tiles.txt')
        return_code = 0
        try:
            touch_file(os.path.join(processing_fol, slide_name))
            cmd = 'python -u save_svs_to_tiles.py {} {} {} >> {}'.format(slide_name, IN_FOLDER, OUT_FOLDER, log_path)
            return_code = os.system(cmd)
            assert return_code == 0     # raise exception if code failed to run
            touch_file(os.path.join(OUT_FOLDER, slide_name, indicator_file))
        except:
            os.system('echo {} >> {}'.format('Failed extracting patches for: ' + slide_name, log_path))
            print('{} - Return code: {}'.format(slide_name, return_code))
            traceback.print_exc(file=sys.stdout)
            #rm_folder(os.path.join(OUT_FOLDER, slide_name))

        touch_file(os.path.join(done_fol, slide_name))
        rm_file(os.path.join(processing_fol, slide_name))


