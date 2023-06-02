import torch
import json
import random
import cv2
import string as STR
from pathlib import Path
import os
import sys
import traceback
from importlib import resources
absolute_path = Path(os.path.dirname(__file__))


class ImageTextDatasetMNT(torch.utils.data.Dataset):
    letters = ' ' + STR.ascii_letters + STR.digits
    clasess = len(letters) + 1

    def __init__(self, image_files, only_synt=False):
        super(ImageTextDatasetMNT, self).__init__()
        self.web2lowerset = json.load(open(absolute_path / 'words.json'))
        self.image_files = image_files
        self.only_synt = only_synt

    def generate_random_string(self):
        types = ['nu', 'wo', 'nuwo']
        ty = random.choice(types)
        string = ''
        if ty == 'nu':
            length = random.randint(1, 10)
            for i in range(length):
                string += str(random.randint(0, 10))
        if ty == 'wo':
            string = random.choice(self.web2lowerset)
        if ty == 'nuwo':
            t_string = list(random.choice(self.web2lowerset))
            indices = random.choices(
                list(range(len(t_string))), k=len(t_string)//3)
            for i in indices:
                t_string[i] = str(random.randint(0, 10))
            string = ''.join(t_string)
        if random.random() > 0.5:
            string = string.upper()

        # string = []
        # for i in range(29):
        #     string.append(random.choice(list(self.letters)))
        # string = ''.join(string)

        return string.replace('-', '')[:10]

    def generate_random_text_image(self):
        string = self.generate_random_string()
        img = torch.zeros((32, 128, 1))
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (0, 24)
        fontScale = 0.6
        color = (255)
        thickness = random.choice(list(range(1, 2)))
        img = cv2.putText(img.numpy(), string, org, font,
                          fontScale, color, thickness, cv2.LINE_AA)
        return torch.tensor(img), string

    def __len__(self):
        return len(self.image_files)

    def pretransform(self, im, resized_sz=(32, 128)):
        sz = im.shape
        new_sz = resized_sz
        ratio = min(new_sz[0]/sz[0], new_sz[1]/sz[1])
        mid_sz = (int(sz[1]*ratio), int(sz[0]*ratio))  # (w, h)
        pd_sz = (new_sz[0]-mid_sz[1], new_sz[1]-mid_sz[0])
        mid_img = cv2.resize(im, mid_sz, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pd_sz[0]/2 - 0.1)
                          ), int(round(pd_sz[0]/2 + 0.1))
        left, right = int(round(pd_sz[1]/2 - 0.1)
                          ), int(round(pd_sz[1]/2 + 0.1))
        final = cv2.copyMakeBorder(
            mid_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return torch.tensor(final).to(torch.float32)

    def __getitem__(self, index):
        if self.only_synt == False:
            try:
                file = self.image_files[index]
                img = cv2.imread(file)
                string = file.split('/')[-1].split('.')[0].split('_')[1]
                img = self.pretransform(img).mean(-1)[..., None]
            except:
                traceback.print_exc()
                print('Error generating data:', file)
                img, string = self.generate_random_text_image()
        else:
            img, string = self.generate_random_text_image()

        img = img/255.0
        img = img.permute([2, 0, 1])

        string = list(string.replace('-', '').replace('.', ''))
        label = []
        max_length = 12
        for i in range(max_length):
            label.append(
                self.letters.index(string[i]) if len(string) > i else -1
            )
        # label = nn.functional.one_hot(
        #     torch.tensor(label), 63).float()
        label = torch.tensor(label)
        label += 1
        return img.to(torch.float32), label, max_length


if __name__ == '__main__':
    dataset = ImageTextDatasetMNT(size=100, only_synt=False)
    img, label, ln = dataset[0]
    print('original images:', img.shape, label, ln)
    dataset = ImageTextDatasetMNT(size=100, only_synt=True)
    img, label, ln = dataset[0]
    print('synthetic images:', img.shape, label, ln)
