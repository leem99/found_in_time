import os
from flask import (Flask, request, redirect, url_for, send_from_directory,\
    render_template)
from werkzeug.utils import secure_filename




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
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>


    <h1>Awesome Time:</h1>
    <h2>Mitch Lee's App to Help You Find the Perfect Wristwatch </h2>
    <h4>Upload new File</h4>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

@app.route('/show/<filename>')
def uploaded_file(filename):
    filename = 'http://127.0.0.1:5000/uploads/' + filename
    return render_template('watch_page.html', filename=filename)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run()
