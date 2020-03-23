import sys
import os
import time

def clear_logs():
    os.system('rm logs/*')

def setup_softlinks():
    os.system('bash 1_start_setup_softlinks_folders.sh')

def submit_jobs_patch_extraction_n_prediction(n_jobs):
    os.system('bash 2_start_patch_extraction.sh {}'.format(n_jobs))
    os.system('bash 3_start_prediction.sh {}'.format(n_jobs))

def count_files(fol):
    fns = [fn for fn in os.listdir(fol) if not fn.startswith('.')]
    return len(fns)

def time_gap_largest(fol):
    files = [f for f in os.listdir(fol) if not f.startswith('.') and not f.endswith('.txt')]
    time_gap = 0
    for fn in files:
        fn_path = os.path.join(fol, fn)
        (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(fn_path)
        time_gap_fn = (time.time() - ctime)/60
        if time_gap_fn > time_gap:
            time_gap = time_gap_fn

    return time_gap


if __name__ == '__main__':
    self_name = os.path.basename(__file__)
    print('Usage: nohup python {} $number_of_loop_run &\n'.format(self_name))
    num_runs = int(sys.argv[1])
    num_jobs = 50

    setup_softlinks()

    log_fol = 'logs'
    for i in range(num_runs):
        print('Running run number: ', i)
        clear_logs()
        submit_jobs_patch_extraction_n_prediction(num_jobs)
        while(1):
            time.sleep(60)
            num_files = count_files(log_fol)
            time_gap = time_gap_largest(log_fol)
            if num_files > num_jobs*2 and time_gap >= 60*2:
                break

    os.system('bash 4_start_cp_heatmaps.sh')
    time.sleep(60)
    while(os.path.exists('~/running_cp_heatmap.txt')):
        time.sleep(30)

    os.system('bash 5_start_gen_json.sh 20')

