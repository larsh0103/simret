import numpy as np
import cv2
from PIL import Image
import simret.api as api
import simret
from torch.utils.data.sampler import RandomSampler
from simret.dataloader import DataWrapper, collater
from torch.utils.data import DataLoader


def visualize_one_image(image,boxes,labels,scores,threshold=0.5):
    print(image.shape)
    print(boxes.shape)
    print(scores)
    for j in range(boxes.shape[0]):
        if scores[j] >threshold:
            bbox = boxes[j]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            print(x1,y1,x2,y2)
            api.draw_caption(image, (x1, y1, x2, y2), "Object")
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
    return image


if __name__ == "__main__":
    model = api.load_model("checkpoints/model-e019.pth")
    cap= cv2.VideoCapture('Pokémon Generations Episode 4 The Lake of Rage.mp4')
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()

        if i % 50 ==0:
            if ret == False:
                break
            print(frame.shape)
            input_image = np.moveaxis(frame.astype(np.uint8),-1,0)
            scores, boxes, labels = api.detect(image = input_image, model=model, device='cuda')
            image = visualize_one_image(frame,boxes,labels,scores)
            cv2.imshow('img', image)
            cv2.waitKey(0)
        i +=1

    

### Todo:
## Make validation set for training
## Make simclr dataset, convert all episodes to frames? do it on the fly?
## Evaluate performance simclr vs no simclr
## Make inference on all frames and produce resized cutouts
## Code Rebase and clean up


#   ann_file = "result.json"
#     img_dir = "pokemon_all"
#     class_names = ["POKEMON"]
#     batch_size = 1
#     dataset_val = DataWrapper(ann_file=ann_file, img_dir=img_dir, class_names=class_names)
#     sampler_val = RandomSampler(data_source=dataset_val)
#     dataloader_val = DataLoader(dataset_val, batch_size=batch_size, sampler=sampler_val, collate_fn=collater)
#     model = api.load_model("checkpoints/model-e015.pth")

#     for data in iter(dataloader_val):
#         image =data['images'][0].cpu().numpy()
#         boxes = data['targets'][0]['boxes']
#         labels = data['targets'][0]['labels']
#         img_path ="pokemon/test/ep-21-frame25004.jpg"
        
#         scores, boxes, labels = api.detect(image = image, model=model)
#         image = np.moveaxis(image.astype(np.uint8),0,-1)
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         print(image.shape)
#         print(boxes.shape)
#         for j in range(boxes.shape[0]):
#             bbox = boxes[j]
#             x1 = int(bbox[0])
#             y1 = int(bbox[1])
#             x2 = int(bbox[2])
#             y2 = int(bbox[3])
#             print(x1,y1,x2,y2)
#             api.draw_caption(image, (x1, y1, x2, y2), "Object")
#             cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
#         cv2.imshow('img', image)
#         cv2.waitKey(0)
        
