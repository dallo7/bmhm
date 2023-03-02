from flask import Flask, render_template, request
import pandas as pd
import joblib

# make our Flask api...
app = Flask(__name__, template_folder="template")

# form urls/...
@app.route("/")
def home():
    return render_template("bmhm.html")

# Accepting form data
@app.route("/prediction", methods=["POST"])
def prediction():
    q = request.form["q"]
    w = request.form["w"]
    e = request.form["e"]
    t = request.form["t"]
    y = request.form["y"]
    u = request.form["u"]
    i = request.form["i"]
    o = request.form["o"]
    h = request.form["h"]
    n = request.form["n"]

    # Creating Out of sample instance
    damp = {"Have you ever experienced a traumatic event?": q, "Country": w,
            "Is there an event in your life you frequently re-live, re-experience or analyze?": e,
            "Is there an event in your life that has resulted in you avoiding particular people, places, "
            "or situations? ": t,
            "Is there a major event in your life that you blame yourself for?  ": y,
            "Do you find it difficult to remember certain features of a particular major event in your life?": u,
            "Is there a major event in your life that makes you feel irritable, aggressive, hyper-aware, "
            "jumpy or even easily startled?": i,
            "Is there an event that has made you withdraw from family, loved ones or even friendships? ": o,
            "Do you have a major life event that has encouraged greater consumption of alcoholic beverages, "
            "smoking or drug use?": h,
            "Does your traumatic event fall in any of the categories below? ": n}

    test = pd.DataFrame(damp, index=[33543])

    model = joblib.load("model")

    pred = model.predict(test)

    if pred[0][0] == pred[0][1] == pred[0][2] == 1:
        pred = "PTSD, Depression and Schizophrenia"
    elif pred[0][0] == pred[0][1] == pred[0][2] == 0:
        pred = "Unknown Condition"
    elif pred[0][0] == 0 and pred[0][1] == 1 and pred[0][2] == 1:
        pred = "Depression and Schizophrenia"
    elif pred[0][0] == 0 and pred[0][1] == 0 and pred[0][2] == 1:
        pred = "Schizophrenia"
    elif pred[0][0] == 0 and pred[0][1] == 1 and pred[0][2] == 0:
        pred = "Depression"
    elif pred[0][0] == 1 and pred[0][1] == 0 and pred[0][2] == 0:
        pred = "PTSD"
    elif pred[0][0] == 1 and pred[0][1] == 0 and pred[0][2] == 1:
        pred = "PTSD and Schizophrenia"
    elif pred[0][0] == 1 and pred[0][1] == 1 and pred[0][2] == 0:
        pred = "PTSD and Depression"
    else:
        pred = "Unstudied Condition"

    # return prediction
    return render_template("bmhm.html", pred="The user demonstrates symptoms likely to suggest they suffer from : {}".format(pred))


# Run this file as the main file...
if __name__ == "__main__":
    app.run(debug=True)
