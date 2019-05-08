from test_methods import *
from unet import *

#method for predicting a given image
def predict_image(image_path,model):
    image = load_image(image_path)
    image = np.reshape(image,(1,)+image.shape)

    result = model.predict(image)
    return result

#method for resizing and predicting on a given image
def predict_from_camera(filename, origin_dir,resized_dir, prediction_dir, model, size=(256,256)):
    image = resize_image(filename,origin_dir,resized_dir, size)
    result = predict_image(resized_dir+filename,model)
    save_result(prediction_dir,result)
    
    
#specify where to store, and where to find image
weights = 'weights.hdf5'
image_name = "image.png"
camera_directory = "from_camera/"
resize_directory = "camera_resized/"
prediction_directory = "camera_prediction/"

#create the model
model = get_unet()
#load pretrained weights
model.load_weights(weights)

predict_from_camera(image_name,camera_directory, resize_directory,prediction_directory,model)