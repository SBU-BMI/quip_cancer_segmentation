import os
import sys


SVS_INPUT_PATH = sys.argv[1]
PATCH_PATH = sys.argv[2]
HEATMAP_TXT_OUTPUT_FOLDER = sys.argv[3]
JSON_OUTPUT_FOLDER = sys.argv[4]

svs_files = set([fn for fn in os.listdir(SVS_INPUT_PATH) if not fn.startswith('.')])
svs_extracting = set([fn for fn in os.listdir(PATCH_PATH) if not fn.startswith('.')])
svs_extracted = set([fn for fn in os.listdir(PATCH_PATH) if os.path.exists(os.path.join(PATCH_PATH, fn, 'extraction_done.txt'))])
prediction = set([fn for fn in os.listdir(PATCH_PATH) if os.path.exists(os.path.join(PATCH_PATH, fn, 'patch-level-cancer.txt'))])

svs_extracting = svs_extracting.intersection(svs_files)
svs_extracted = svs_extracted.intersection(svs_files)
prediction = prediction.intersection(svs_files)

heatmap_txt = set([fn.split('prediction-')[-1] for fn in os.listdir(HEATMAP_TXT_OUTPUT_FOLDER) if fn.startswith('prediction-')])
heatmap_json = set([fn.split('heatmap_')[-1].split('.json')[0] for fn in os.listdir(JSON_OUTPUT_FOLDER) if fn.startswith('heatmap_')])

print('Patch extraction: number of WSIs: {}; number of WSIs extracting: {}; number of WSIs extracted: {}'.format(len(svs_files), len(svs_extracting), len(svs_extracted)))
print('Prediction: number of WSIs extracted: {}; number of WSIs predicted: {}'.format(len(svs_extracted), len(prediction)))
print('Copy heatmap_txt: number WSIs predicted: {}; number heatmap_txt copied: {}'.format(len(prediction), len(heatmap_txt)))
print('Generate heatmap_json: number heatmap_txt: {}; number jsons: {}'.format(len(heatmap_txt), len(heatmap_json)))

print('\nRemaining WSIs not extracted: ', svs_files.difference(svs_extracted))
print('\nRemaining WSIs not predicted: ', svs_extracted.difference(prediction))

