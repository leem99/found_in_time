import os
from flask import (Flask, request, redirect, url_for, send_from_directory,\
    render_template)
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from keras.models import load_model, Model
from keras.applications.inception_v3 import InceptionV3

#----------  Open Model --------------------#
fvecs = pd.read_csv('all_watch_info.csv')


#---------- URLS AND WEB PAGES -------------#
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg','JPG'])

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
            global rec_watch
            rec_watch = 'https://www.prestigetime.com/images/watches/214270_Black_Luminous.jpg'
            global rec_list
            rec_list = ['https://www.prestigetime.com/images/watches/311.33.42.30.01.001.jpg',
                'https://www.prestigetime.com/images/watches/214270_Black_Luminous.jpg']
            
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template('home_page.html') 

@app.route('/show/<filename>')
def uploaded_file(filename):
    filename = 'http://127.0.0.1:5000/uploads/' + filename
    return render_template('watch_page.html', filename=filename,rec_list=rec_list)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run()
