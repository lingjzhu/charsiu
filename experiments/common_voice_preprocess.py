from datasets import load_dataset
import soundfile as sf
from tqdm.contrib.concurrent import process_map
import re
import subprocess
from tqdm import tqdm

common_voice_train = load_dataset("common_voice", "en", split="train+validation",cache_dir='/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/asr')

file_pairs = [(i['path'],re.search(r'(.*?)\.mp3',i['path']).group(1)+'.wav') for i in tqdm(common_voice_train)]

def convert_and_resample(item):
    command = ['sox', item[0],'-r','16000',item[1]]
    subprocess.run(command)
    
r = process_map(convert_and_resample, file_pairs, max_workers=8, chunksize=1)
