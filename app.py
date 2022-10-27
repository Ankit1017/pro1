from flask import Flask,request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("home.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    # print(int_features)
    # print(final)
    prediction=model.predict(final)
    print(prediction)
    output='{0:.{1}f}'.format(prediction[0], 2)
    if output>str(1):
        return render_template('home.html',pred='Your Friend is in Danger.\nProbability of Ch*tiyaness is 1',bhai="kuch karna hain iska ab?")
    elif output>str(0.5):
        return render_template('home.html',pred='Your Friend is in Danger.\nProbability of Ch*tiyaness is {}'.format(output),bhai="kuch karna hain iska ab?")
    elif output<str(0):
        return render_template('home.html',pred='Your Friend is not in Danger.\nProbability of Ch*tiyaness is 0',bhai="Your Friend is Safe for now")
    else:
        return render_template('home.html',pred='Your Friend is not in Danger.\nProbability of Ch*tiyaness is {}'.format(output),bhai="Your Friend is Safe for now")


if __name__ == '__main__':
    app.run(debug=True)
