Main model training tool with ability to retrain, train anew and resume from last epoch - ```model-train.py```

Python version 3.13

## Script was done and tested on Windows, the paths and functions might not work correctly in Linux and Mac.

## Need to create:
configs/project_config.yaml content:

```yaml
roboflow:
  roboflow_api_key: ""
  roboflow_workspace: ""
  roboflow_project: ""


project_settings:
  set_dataset: ""
  model_train: ""

CVAT:
  url: "http://localhost:8080"
  username: ""
  password: ""

utils:
  r34_user_id: ""
  r34_api_key: ""

  reddit_user_agent: ""
  reddit_client_id: ""
  reddit_client_secret: ""
  reddit_username: ""
  reddit_password: ""


autolabel:
  segmentation_model: "seg.pt"
  detection_model: "det.pt"
  label_folder: ""
  sam_model: "sam2.1_l.pt"

google_scraper:
  image_query: ""
```

configs/train_conf.yaml content:
```yaml
batch: -1  # 16 default, -1 autodetects
device: 0  # use GPU for training
cache: 'disk'  # Use 'disk' for cache - True: RAM can missmatch
save: True  # Save the models during training so you can resume later
save_period: 10 # Save every 10 epochs
optimizer: 'auto'
amp: True

# Augmentation Settings and Hyperparameters
erasing: 0.4
auto_augment: 'autoaugment'
```




## utils folder has some utlities used for various things:

### CVAT/auto-annotate.py 

After specifying use_folder in ```config.autolabel.label_folder``` it will go through that path, check the folder/images folder and list the images and movies, if it's a movie or a gif, then it will extract the frames from it( every 5th frame for 30fps and every 10th if 60fps or higher- you can change it in similarity_threshold, stride_high_fps, stride_low_fps), move out the ```video/gif to folder/processed_vids``` and then create a list from all the files and compare it against your localhost CVAT instance index and if duplicate is found then it will be moved out to ```folder/found_duplicates```, the frames won't be saved unless they are different enough( specified in similarity_threshold) and will start auto-annotating with the models specified in: ```SAM_model + detection_model / segmentation_model``` depending how you want to annotate. 

After the annotation is completed, script will ask if you want to go into the folder and do anything in it- I check if I want to keep all the images or remove something by hand, if you answer yes then it will open the file explorer to that path and will wait till you pass continue to it. 



After that, the data will be prepared in CVAT1.1 format in a foldername.zip ready to import into CVAT as a new dataset. 


### cvat_prep.py
After exporting data from CVAT, and setting the variable use_dataset = config.project_settings.set_dataset . unpack the data.yaml and images + labels in it and then the script will prepare the data for yolo model training, it only moves the pairs( image + label with data) so if the image is missing label or label missing image, it won't be moved. 

It also splits the dataset into 3: train, val, test and you can change the split inside 
```SPLIT_RATIO = (0.8, 0.1, 0.1) - 80% of data goes to train, 10% to val and 10% to test.``` Currently it only checks the train.txt so if you have data spread in CVAT in different categories like validation as well, it will need to be added ( probably will add soon). Then during model training that data will be just used to train.


## cvat_utils.py
Simple script that I needed at one point, it simply checks the CVAT database if the filename exists and prints which project is it in and which task and frame position so you can easily find it. 



## Google-image-scrapper.py
Simple google image query scrapper, query set in:
```
image_query = config.google_scraper.image_query 
```

It checks for duplicates and doesn't download if present, runs in async, but sometimes can still be slow. Change ```max_images``` if you want to limit it below 100.


## r34-scrape.py

Pretty much what the name suggests, it scrapes images from rule34 website, you set the main tag in ```add_tags list, TAGS``` list is a list of exclusions. You need to create account and create api key for it to work and set it in: 
```
client.api_key = config.utils.r34_api_key
client.user_id = config.utils.r34_user_id
```

It also provides a link in the printout if you want to check it out what is it downloading or test.



## reddit_image_scrapper.py & redgifs_downloader.py 
Subreddit image, gif and video scrapper, it's used from CMD rather than run from within itself, sample run:


```py
python reddit_image_scraper.py --subs "cats,birds,dogs," --sort hot --limit 500
```

Arguments you can use:
```
    --subs: Comma-separated list of subreddits to scrape
    --subs-file: Path to a file containing one subreddit per line
    --limit: Maximum number of posts to scrape per subreddit
    --sort: Sorting method for posts (hot, new, top, rising, controversial)
    --time_filter: Time filter for posts (all, day, hour, month, week, year)
    --out: Output directory for downloaded media
    --skip-gifs: Skip downloading GIFs
```
Note: you can choose a list of subreddits and just separate them with comma, and no space after. It will download data to reddit_images/subbredit_name/ so the data will be separate for every query.

redgifs_downloader.py needs to be in the same dir, it takes care of downloading e data from there- it runs automatically when you start reddit image scrapper.

For this you will also need to setup the reddit authorized app and grab name and api key : https://www.reddit.com/prefs/apps , set it up in the praw.ini
```ini
[scraperr]
; this is your 14 character personal use script
client_id=""
; this is your 27 character secret
client_secret=""
; this is the name you gave your application
user_agent=""
; this is username for the reddit account the app was created with
username=""
; password for the account
password=""
```


## label-studio: clean-json.py
A simple script that takes the json export from label-studio and images, then cleans up the json so it can be used for data preparation for yolo model training( label studio locally just gives the weird path in json which is not accesible by yolo ( /data/local-files/?d= ). This probably can be moved to the main LS-prep.py and will work right off the bat for all-in-one solution.



## label-studio: LS-prep.py
Script used to prepare dataset for YOLO training from json export in Label Studio. 


Put the ```images folder and data.json``` in same folder and name it, then set that name in ```PROJECT_NAME```.
You can set the split of data between other folders: ```val, test and train```:
```py
VAL_FRACTION = 0.1   # 10% validation
TEST_FRACTION = 0.1  # 10% test
```

Make sure to set:
```
PROJECT_NAME
EXPORT_FILE
IMAGES_DIR
OUTPUT_DIR
```
After the run is completed, the dataset should be ready to train the yolo model.
