This directory has

the integer corresponding to each word (the same in train and test)
   -- voc-top-5000.txt
      note that "words" can contain commas and other odd symbols, but no whitespace

code to visualize the images
  --  show_images.m

subdiretories with the training (n=4998) and test sets (n=4997)

1) the outcome (1=female, 0=male) - only for the training
   -- genders_*.txt
2) how often each user used each of the 5,000 most frequent words
   -- words_*.txt
3) the raw images 100*100*RGB
   -- images_*.txt
4) a set of extracted features for each user's image
   image_features_*.txt

wher '*' is either train or test





