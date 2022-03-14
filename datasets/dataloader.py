import os

import torch
import csv
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from scipy import signal
import random
from audio_io import load_audio_av, open_audio_av

class GetAudioVideoDataset_Train(Dataset):
    def __init__(self, args, transforms=None):
        
        self.audio_path = f"{args.train_data_path}/audio/"
        self.video_path = f"{args.train_data_path}/frames/"

        # List directory
        audio_files = {fn.split('.wav')[0] for fn in os.listdir(self.audio_path) if fn.endswith('.wav')}
        image_files = {fn.split('.jpg')[0] for fn in os.listdir(self.video_path) if fn.endswith('.jpg')}
        avail_files = audio_files.intersection(image_files)
        print(f"{len(avail_files)} available files")

        # Subsample if specified
        if args.trainset.lower() in {'vggss', 'flickr'}:
            pass    # use full dataset
        else:
            subset = set(open(f"{args.train_data_path}/{args.trainset}.txt").read().splitlines())
            avail_files = avail_files.intersection(subset)
            print(f"{len(avail_files)} valid subset files")

        avail_files = sorted(list(avail_files))
        self.audio_files = sorted([dt+'.wav' for dt in avail_files])
        self.video_files = sorted([dt+'.jpg' for dt in avail_files])

        self.imgSize = args.image_size 

        self.transforms = transforms
        # initialize video transform
        self._init_atransform()
        self._init_transform()

        print('train:', len(self.video_files))
        self.count = 0

    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        self.img_transform = transforms.Compose([
            transforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC),
            transforms.RandomCrop((self.imgSize, self.imgSize)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])      

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.0], std=[12.0])])
#   

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def __len__(self):
        # Consider all positive and negative examples
        return len(self.video_files)  # self.length

    def getitem(self, idx):
        file = self.video_files[idx]

        # Image
        frame = self.img_transform(self._load_frame(self.video_path + file[:-3] + 'jpg'))

        # Audio
        # samples, samplerate = sf.read(self.audio_path + file[:-3]+'wav')
        # samples, samplerate = librosa.load(self.audio_path + file[:-3]+'wav')

        T = 3
        audio_ctr = open_audio_av(self.audio_path + file[:-3]+'wav')
        audio_dur = audio_ctr.streams.audio[0].duration * audio_ctr.streams.audio[0].time_base
        audio_ss = max(float(audio_dur)/2 - T/2, 0)
        samples, samplerate = load_audio_av(container=audio_ctr, start_time=audio_ss, duration=T)
        samples = samples[0]

        # repeat if audio is too short
        if samples.shape[0] < samplerate * T:
            n = int(samplerate * T / samples.shape[0]) + 1
            samples = np.tile(samples, n)
        resamples = samples[:samplerate*T]

        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=274)
        spectrogram = np.log(spectrogram + 1e-7)
        spectrogram = self.aid_transform(spectrogram)

        return frame, spectrogram, resamples, file

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception:
            return self.getitem(random.sample(range(len(self)), 1)[0])


class GetAudioVideoDataset_Test(Dataset):

    def __init__(self, args, transforms=None):

        if args.testset == 'flickr':
            testcsv = 'metadata/flickr_test.csv'
        elif args.testset == 'vggss':
            testcsv = 'metadata/vggss_test.csv'
        elif args.testset == 'vggss_heard':
            testcsv = 'metadata/vggss_heard_test.csv'
        elif args.testset == 'vggss_unheard':
            testcsv = 'metadata/vggss_unheard_test.csv'
        else:
            raise NotImplementedError

        self.audio_path = args.test_data_path + 'audio/'
        self.video_path = args.test_data_path + 'frames/'

        audio_files = {fn.split('.wav')[0] for fn in os.listdir(self.audio_path)}
        image_files = {fn.split('.jpg')[0] for fn in os.listdir(self.video_path)}
        avail_files = audio_files.intersection(image_files)
        data = [item[0] for item in csv.reader(open(testcsv))]
        data = [dt for dt in data if dt in avail_files]

        self.imgSize = args.image_size 

        self.transforms = transforms
        # initialize video transform
        self._init_atransform()
        self._init_transform()

        #  Retrieve list of audio and video files
        self.video_files = []

        for item in data[:]:
            self.video_files.append(item)
        print('test:', len(self.video_files))
        self.count = 0

    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.imgSize, self.imgSize), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])            

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.0], std=[12.0])])

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def __len__(self):
        # Consider all positive and negative examples
        return len(self.video_files)  # self.length

    def getitem(self, idx):
        file = self.video_files[idx]

        # Image
        frame = self.img_transform(self._load_frame(self.video_path + file + '.jpg'))

        # Audio
        # samples, samplerate = sf.read(self.audio_path + file[:-3]+'wav')
        # samples, samplerate = librosa.load(self.audio_path + file[:-3]+'wav')
        T = 3
        audio_ctr = open_audio_av(self.audio_path + file+'.wav')
        audio_dur = audio_ctr.streams.audio[0].duration * audio_ctr.streams.audio[0].time_base
        audio_ss = max(float(audio_dur)/2 - T/2, 0)
        samples, samplerate = load_audio_av(container=audio_ctr, start_time=audio_ss, duration=T)
        samples = samples[0]

        # repeat if audio is too short
        if samples.shape[0] < samplerate * T:
            n = int(samplerate * T / samples.shape[0]) + 1
            samples = np.tile(samples, n)
        resamples = samples[:samplerate*T]

        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples,samplerate, nperseg=512,noverlap=274)
        spectrogram = np.log(spectrogram + 1e-7)
        spectrogram = self.aid_transform(spectrogram)

        return frame, spectrogram, resamples, file

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception:
            return self.getitem(random.sample(range(len(self)), 1)[0])


def inverse_normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    inverse_mean = [-0.485/0.229,-0.456/0.224,-0.406/0.225]
    inverse_std = [1.0/0.229, 1.0/0.224, 1.0/0.225]
    tensor = transforms.Normalize(inverse_mean, inverse_std)(tensor)
    return tensor



