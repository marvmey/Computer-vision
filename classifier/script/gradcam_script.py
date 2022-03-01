
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import os
import cv2
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model



#  VARIABLES

model_path = ''
dir_img_grad_cam =
dir_dataset =
target_size =
img_size =
last_conv_layer_name =
prefix_to_remove =


#  LOAD AND CHECK THE MODEL
model = load_model(model_path)
model.summary()


# # **Creer dataset **


dir = dir_dataset
label = []
path = []


for dirname, _,filenames in os.walk(dir):
    for filename in filenames:
        if os.path.splitext(filename)[1]=='.png':
            if dirname.split()[-1]!='GT':
                label.append(os.path.split(dirname)[1])
                path.append(os.path.join(dirname,filename))

df = pd.DataFrame(columns=['path','label'])
df['path']=path
df['label']=label
df['label']=df['label'].astype('category')
df['label'].value_counts()


X_train, X_test=train_test_split(df, test_size=0.99, random_state=42)

testGen =ImageDataGenerator( preprocessing_function= preprocess_input)
X_test_img =testGen.flow_from_dataframe(dataframe=X_test, x_col='path', y_col='label',
                                        class_mode='categorical', color_mode='rgb',
                                        batch_size=32, shuffle=False,target_size = target_size)



# PREDICT ON ALL DATASET


predict_x=model.predict(X_test_img)
classes_x=np.argmax(predict_x,axis=1)
pred_df=X_test.copy()
labels={}
for l,v in X_test_img.class_indices.items():
    labels.update({v:l})
pred_df['pred']=classes_x
pred_df['pred']=pred_df['pred'].apply(lambda x: labels[x])



# CREATE GOOD PRED AND BAD PRED DATASET

bad_labels = []
bad_preds = []
bad_paths = []
good_labels = []
good_preds = []
good_paths = []

for i in pred_df.index:

    if (pred_df.loc[i, 'label']) != (pred_df.loc[i, 'pred']):
        bad_labels.append(pred_df['label'][i])
        bad_preds.append(pred_df['pred'][i])
        bad_paths.append(path[i])
    else:
        
        good_labels.append(pred_df['label'][i])
        good_preds.append(pred_df['pred'][i])
        good_paths.append(path[i])

df_bad_labels = pd.DataFrame({'label': bad_labels })
df_bad_preds = pd.DataFrame({'pred': bad_preds})
df_bad_paths = pd.DataFrame({'path': bad_paths})

df_good_labels = pd.DataFrame({'label':good_labels})
df_good_preds = pd.DataFrame({'pred': good_preds})
df_good_paths = pd.DataFrame({'path':good_paths})

final_bad_pred = pd.concat([df_bad_paths, df_bad_labels, df_bad_preds], axis=1)
final_good_pred = pd.concat([df_good_paths,df_good_labels,df_good_preds], axis = 1)




# # Preparer fonction **GRADCAM**

def get_img_array(img_path, size):
    
    img = keras.preprocessing.image.load_img(img_path, target_size=size)   
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
   
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()





def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)
    display(Image(cam_path))




def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text 


def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string




# APPLY GRADCAM ON BAD AND GOOD PRED

for i in final_bad_pred.index:
    img_path = final_bad_pred.loc[i,'path']
    a = img_path.replace('/','_')
    x = final_bad_pred.loc[i,'pred']
    b = remove_prefix(a, prefix_to_remove)
    c = f'GRAD_CAM_NONOK-[pred:{x}]-{b}'
    cam_path = f'{dir_img_grad_cam}{c}' 
    img_array = preprocess_input(get_img_array(img_path, size=img_size))
    model.layers[-1].activation = None
    preds = model.predict(img_array)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    save_and_display_gradcam(img_path,heatmap, cam_path  )



for i in final_good_pred['path']:
    img_path = i
    i = i.replace('/','_')
    i = remove_prefix(i,prefix_to_remove)
    i = f'GRAD_CAM-{i}'
    cam_path = f'{dir_img_grad_cam}{i}' 
    img_array = preprocess_input(get_img_array(img_path, size=img_size))
    model.layers[-1].activation = None
    preds = model.predict(img_array)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    save_and_display_gradcam(img_path,heatmap, cam_path  )

