# Unsupervised people separator from photos

Can be used as a huge time saving tools in labeling images by face eg. creating a dataset of face recognition.<br><br>
Seperates images of different people in different folders by unsupervised learning so that you can send all of the photos if your friend who keeps on saying "ऐ मेरो सबै फोतो पठाईदे ह तेरो मोबाइलको।" (Send all my photos to your on phone.) ** ```Literal google translate```
<br><br>
Best works in one person per image


## Methodology

Uses face detection modules to extract only face then gets embeddings from the cropped face. These embeddings are then used to do unsupervised clustering through which large amount of images are clustered and seperated according to people in it.

## To run

Store all the image in folder of name of your choice in the main directory.

Download models folder from [this link](https://drive.google.com/open?id=1CfO8behECt2Uey9TIb5rsPrt-cUmo8Ou)

Then run
```bash
python3 unsupervised.py
```

