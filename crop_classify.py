from PIL import Image
from pandas import read_csv
import os
import subprocess


def read_data_find_crop_area(data_file='prediction_george.csv',):

    data = read_csv(data_file,skipinitialspace=True)
    highest_propability_obs = data['Prediction'].argmax()
    observation = data.loc(highest_propability_obs)[0]
    area = (observation['Left'], observation['Top'], observation['Right'], observation['Bottom'])
    
    return area


def crop_image(area):

    img = Image.open("predictions.jpg")
    cropped_img = img.crop(area)
    cropped_img.show()
    cropped_img.save('Cropped.jpg')
    img.close()
    
    return

def archive_observations(data_file='prediction_george.csv', archive_file='archive_observations.csv'):

    f = open(archive_file,'a')
    data = read_csv(data_file,skipinitialspace=True)
    f.write(data.to_csv(header=None,index=False))
    f.close() 

    return 

def remove_observations(data_file='prediction_george.csv'):

    f = open(data_file,'w')
    f.write('Object,Prediction,Left,Top,Right,Bottom\n')
    f.close()

    return 


def execute_classification(poet_path='/Users/giorgoschantzialexiou/Repositories/tensorflow-for-poets-2'):

    classification_command = "python -m scripts.label_image \
     --graph=tf_files/retrained_graph_george.pb \
     --image=/Users/giorgoschantzialexiou/Repositories/design_of_internet_services/dataset/superdry/cropped_back.jpg \
     --labels=tf_files/retrained_labels_george.txt".split()
    os.chdir(poet_path)
    classification = subprocess.check_output(classification_command)
    f = open('/Users/giorgoschantzialexiou/Repositories/darknet/classify.txt','w')
    results = classification.split('(1-image):')[1]
    f.write(results)
    f.close()
    return 

if __name__=='__main__':

    data_file = 'prediction_george.csv'
    archive_file = 'archive_observations.csv'
    poet_path = '/Users/giorgoschantzialexiou/Repositories/tensorflow-for-poets-2'
    
    crop_image(read_data_find_crop_area(data_file=data_file))
    archive_observations(data_file=data_file, archive_file=archive_file)
    remove_observations(data_file=data_file)
    execute_classification()

