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

def clean_done_fol(done_fol, out_fol, *indicator_files):
    done_fns = list_files(done_fol)
    out_fns = list_files(out_fol)
    for fn in done_fns:
        if not all([is_path_exists(out_fol, fn, indicator_file) for indicator_file in indicator_files]):
            rm_file(os.path.join(done_fol, fn))

    for fn in out_fns:
        if all([is_path_exists(out_fol, fn, indicator_file) for indicator_file in indicator_files]) and not is_path_exists(done_fol, fn):
            touch_file(os.path.join(done_fol, fn))

if __name__ == '__main__':
    IN_FOLDER = sys.argv[1]
    CNN_MODEL = sys.argv[2]
    LOG_OUTPUT_FOLDER = sys.argv[3]
    cnn_file='patch-level-cancer.txt'
    color_file = 'patch-level-color.txt'
    indicator_file = 'extraction_done.txt'

    done_fol = 'done'
    processing_fol = 'processing'
    create_fol(done_fol)
    create_fol(processing_fol)
    clean_done_fol(done_fol, IN_FOLDER, cnn_file, color_file)

    time.sleep(random.randint(100, 1000)/100.0)  # wait for 1 --> 10s to avoid concurrency
    start_time = time.time()
    while(1):
        clean_files(processing_fol, 60)
        elapsed_time = time.time() - start_time
        if elapsed_time > 100*60: # require at least 20mins left to start processing a new WSI
            exit(0)

        svs_done = set(list_files(done_fol))
        svs_processing = set(list_files(processing_fol))
        svs_extracting = set(list_files(IN_FOLDER))
        svs_extracted = set([fn for fn in list_files(IN_FOLDER) if is_path_exists(IN_FOLDER, fn, indicator_file)])
        svs_remaining = svs_extracted.difference(svs_done.union(svs_processing))
        print('{}- len of done: {}, processing: {}, extracing: {}, extracted: {}, remaining slides: {}'.format(time.ctime(), len(svs_done), len(svs_processing), len(svs_extracting), len(svs_extracted), len(svs_remaining)))

        if len(svs_done) + len(svs_processing) >= len(svs_extracting):
            print(time.ctime(), '- Done prediction')
            exit(0)

        if len(svs_remaining) == 0:
            time.sleep(30)
            print(time.ctime(), '- waiting for new WSI...')
            continue

        svs_remaining = list(svs_remaining)
        random.shuffle(svs_remaining)
        slide_name = svs_remaining[0]
        print(time.ctime(), '- Start processing slide: ', slide_name)
        log_cnn = os.path.join(LOG_OUTPUT_FOLDER, 'log.cnn.txt')
        log_color = os.path.join(LOG_OUTPUT_FOLDER, 'log.color.txt')
        try:
            touch_file(os.path.join(processing_fol, slide_name))
            cmd_color = 'nohup python -u color_stats.py {} {} >> {} &'.format(os.path.join(IN_FOLDER, slide_name), color_file, log_color)
            cmd_cnn = 'python -u pred.py {} {} {} {} >> {}'.format(os.path.join(IN_FOLDER, slide_name), '_', cnn_file, CNN_MODEL, log_cnn)
            print(cmd_color)
            print(cmd_cnn)

            color_code = os.system(cmd_color)
            cnn_code = os.system(cmd_cnn)

            print(color_code, cnn_code)
            assert color_code == 0
            assert cnn_code == 0
        except:
            if color_code:
                os.system('echo {} >> {}'.format('----------Failed compute color for: ' + slide_name, log_color))
            if cnn_code:
                os.system('echo {} >> {}'.format('----------Failed prediction for: ' + slide_name, log_cnn))
            traceback.print_exc(file=sys.stdout)

        touch_file(os.path.join(done_fol, slide_name))
        rm_file(os.path.join(processing_fol, slide_name))


