import torchvision.transforms as T


class TwoCropsWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        query_image = self.transform(x)
        key_image = self.transform(x)
        return [query_image, key_image]


class MoCoDataAugmentation:
    def __init__(self):
        self.cropper = T.RandomResizedCrop(224, scale=(0.2, 1))
        self.flip_and_color_jitter = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(0.4, 0.4, 0.4, 0.4),
            T.RandomGrayscale(p=0.2),
        ])
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __call__(self, x):
        return self.normalize(self.flip_and_color_jitter(self.cropper(x)))
