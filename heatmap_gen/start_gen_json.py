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

def list_files(fol, template=''):
    files = [f for f in os.listdir(fol) if not f.startswith('.') and template in f]
    return files

def is_path_exists(*args):
    path = '/'.join(args)
    return os.path.exists(path)

def clean_done_fol(done_fol, out_fol):
    done_fns = set(list_files(done_fol))
    out_fns = set([f.split('heatmap_')[-1].split('.json')[0] for f in list_files(out_fol, 'heatmap')])
    for fn in done_fns:
        if fn not in out_fns:
            rm_file(os.path.join(done_fol, fn))
    for fn in out_fns:
        if fn not in done_fns:
            touch_file(os.path.join(done_fol, fn))

if __name__ == '__main__':
    print('start start_gen_json.py')

    IN_FOLDER = sys.argv[1]     # heatmap_txt
    SVS_INPUT_PATH = sys.argv[2]
    HEATMAP_VERSION = sys.argv[3]
    LOG_OUTPUT_FOLDER = sys.argv[4]
    OUT_FOLDER = 'json'

    done_fol = 'done'
    processing_fol = 'processing'
    create_fol(done_fol)
    create_fol(processing_fol)
    clean_done_fol(done_fol, OUT_FOLDER)

    time.sleep(random.randint(100, 1000)/100.0)  # wait for 1 --> 10s to avoid concurrency
    start_time = time.time()
    while(1):
        elapsed_time = time.time() - start_time
        if elapsed_time > 100*60: # require at least 20mins left to start processing a new WSI
            exit(0)

        svs_done = set(list_files(done_fol))
        svs_processing = set(list_files(processing_fol))
        svs_all = set([f.split('prediction-')[-1] for f in list_files(IN_FOLDER, 'prediction-')])
        svs_remaining = svs_all.difference(svs_done.union(svs_processing))
        print('{} - done:{}, processing:{}, all:{}, remaining:{}'.format(time.ctime(), len(svs_done), len(svs_processing), len(svs_all), len(svs_remaining)))
        if len(svs_remaining) == 0:
            exit(0)

        clean_files(processing_fol, 60)

        svs_remaining = list(svs_remaining)
        random.shuffle(svs_remaining)
        slide_name = svs_remaining[0]
        log_path = os.path.join(LOG_OUTPUT_FOLDER, 'log.gen_json.txt')
        cmd = 'python -u gen_json_multipleheat.py {} {} {} lym 0.5 necrosis 0.5 >> {}'.format(os.path.join(IN_FOLDER, 'prediction-' + slide_name), HEATMAP_VERSION, SVS_INPUT_PATH, log_path)
        try:
            touch_file(os.path.join(processing_fol, slide_name))
            return_code = os.system(cmd)
            assert return_code == 0     # raise exception if code failed to run
        except:
            os.system('echo {} >> {}'.format('Failed generating json file for: ' + slide_name, log_path))
            print(cmd)
            traceback.print_exc(file=sys.stdout)

        touch_file(os.path.join(done_fol, slide_name))
        rm_file(os.path.join(processing_fol, slide_name))


