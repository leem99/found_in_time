import os
from flask import (Flask, request, redirect, url_for, send_from_directory,\
    render_template)
from werkzeug.utils import secure_filename

import pandas as pd
import numpy as np
from PIL import Image

from sklearn.metrics.pairwise import cosine_similarity
from keras.models import load_model, Model
from keras.applications.inception_v3 import InceptionV3

#---------- OPEN DATA -----------------------#

# Data for Pre-Computed Model Feature Vectors
watch_names = pd.read_csv('inceptionv3_raw_feature_vectors_names.csv',index_col=0)
watch_names['file_name'] = [x.split('/')[-1].replace('.jpg','')  for x in watch_names['file_name']]
watch_names = watch_names[:-1]
f_vecs = np.load('inceptionv3_raw_feature_vectors.npy')
f_vec_df = pd.DataFrame(f_vecs, index=[watch_names['file_name']])

# Descriptive Data Regarding Each watch
watch_df = pd.read_csv('all_watch_info_with_indicators.csv')
url_dict = dict(zip(watch_df.image_name, watch_df.image_url))

#---------- OPEN Model ----------------------#
watch_model = load_model('inceptionv3_raw_2048.h5')

#---------- Helper Functions  ---------------#
def prepare_image(image_name):
    im_watch = Image.open(image_name)
    im_watch = im_watch.resize((299,299))
    im_watch_array = np.array(im_watch)/255
    im_watch_array = np.expand_dims(im_watch_array, axis=0)
    
    return im_watch_array

def get_similar(watchs, n=None):
    """
    calculates which watchs are most similar to the watchs provided. Does not return
    the watchs that were provided
    
    Parameters
    ----------
    watchs: list
        some watchs!
    
    Returns
    -------
    ranked_watchs: list
        rank ordered watchs
    """
    watchs = [watch for watch in watchs if watch in dist_df.columns]
    watchs_summed = dist_df[watchs].apply(lambda row: np.sum(row), axis=1)
    watchs_summed = watchs_summed.sort_values(ascending=False)
    ranked_watchs = watchs_summed.index[watchs_summed.index.isin(watchs)==False]
    ranked_watchs = ranked_watchs.tolist()
    if n is None:
        return ranked_watchs
    else:
        return ranked_watchs[:n]

def make_rec(img,n_recs):
    img_fvec = watch_model.predict(img)
    similarities = cosine_similarity(img_fvec,np.array(f_vecs)) 
    top_matches = np.array(similarities).argpartition(-n_recs)[0][-n_recs:]
    recommend_names = f_vec_df.index[top_matches]
    return recommend_names
    


#---------- URLS AND WEB PAGES -------------#
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg','JPG'])
uploaded_files = []

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            global uploaded_files
            uploaded_files.append(filename)
            
            img = prepare_image('uploads/'+str(filename)) 
            n_recs = 6
            
            recommend_names = make_rec(img,n_recs)

            #global rec_list
            global rec_dict
            rec_dict = dict()
            
            rec_list = []
            for rec in recommend_names:
                rec_list.append(
                    [rec,'http://127.0.0.1:5000/static/images/'+rec+'.jpg'])

            rec_dict[filename] = rec_list    
                
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template('home_page.html') 

@app.route('/show/<filename>')
def uploaded_file(filename):
    
    if filename in rec_dict:
        rec_list=rec_dict[filename]
    else:
        # If pred doesn't exist make new pred
        img = prepare_image(
            'static/images/'+filename+'.jpg') 
        n_recs = 6
        recommend_names = make_rec(img,n_recs)
        rec_list = []
        for rec in recommend_names:
            rec_list.append(
                [rec,'http://127.0.0.1:5000/static/images/'+rec+'.jpg'])
        rec_dict[filename] = rec_list
    
    # Check if Uploaded Files
    if filename in uploaded_files:
        filename = 'http://127.0.0.1:5000/uploads/' + filename
    else:    
        filename = 'http://127.0.0.1:5000/static/images/'+ filename + '.jpg'
    return render_template('watch_page.html', filename=filename,rec_list=rec_list)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run()
