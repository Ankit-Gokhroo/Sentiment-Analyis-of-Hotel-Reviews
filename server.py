# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:55:06 2019

@author: user
"""

from flask import Flask, render_template, request, url_for, redirect
from sentimental_analysis import sentiments

app = Flask(__name__)

# @app.route("/home")
# def home():
#     return render_template("index.html", response=None)

@app.route("/analysis",methods=["GET","POST"])
def result():
	if request.method == "GET":
		return render_template("index.html", response=None)
	else:
	    data = request.form["card"]
	    output = sentiments(data)
	    return render_template("index.html",response=True,output=output)

if __name__ == "__main__":
    app.run(debug=True)
