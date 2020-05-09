import random
from itertools import chain
from tkinter import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

window = Tk()
window.resizable(0, 0)
window.attributes("-topmost", True, "-alpha", 0.95)
window.title("SB: Corona Search")
window.configure(bg="#ffffff")


def startsearch(e):
    """

    data exctraction

    """

    Hospitalized_df = pd.read_csv("corona_data\\newdata\\total_cases.csv")
    death_df = pd.read_csv("corona_data\\newdata\\total_deaths.csv")

    # insert country
    country = (ent.get().lower()).capitalize()

    # death per day total
    Death_graph = death_df[country].fillna(method='pad')

    # country to search
    Hospitalized_graph = Hospitalized_df[country].fillna(method='pad')

    # hospitalized data to represent
    Hospitalized_data = []
    Hospitalized_data.append(Hospitalized_graph)
    Hospitalized_data = np.reshape(Hospitalized_data, (1, -1))
    Hospitalized_data = list(chain.from_iterable(Hospitalized_data))

    # everyday new case genrator
    new_case = [int(Hospitalized_data[0])]

    for i in range(1, len(Hospitalized_df)):
        new_case.append(int(Hospitalized_data[i]) - int(Hospitalized_data[i - 1]))

    newcase_df = pd.DataFrame(np.array(new_case).reshape(-1, 1), columns=["New"])

    """
    
    Mutation Rate calculation
    
    """

    k = 0
    j = 0
    mutation_rate = [int(Hospitalized_data[0])]
    # print("i k i i-1 rate")
    # print("2 ",new_case[0],"1: ", mutation_rate[0])
    for i in range(1, len(Hospitalized_data)):
        # print(i, k, new_case[i], new_case[i - 1], mutation_rate[i - 1])
        if (Hospitalized_data[i - 1] == 0 and Hospitalized_data[i] == 0):
            mutation_rate.append(0)
            # print(i+2, new_case[i], "0: ", mutation_rate[i])
        elif (Hospitalized_data[i] != 0 and Hospitalized_data[i - 1] == 0):
            if (k == 0):
                mutation_rate.append(0)
            else:
                mutation_rate.append(((Hospitalized_data[i] / (Hospitalized_data[k]))))
            # print(i+2, new_case[i], "2: ", mutation_rate[i])
        else:

            mutation_rate.append(((Hospitalized_data[i] / (Hospitalized_data[i - 1]))))
            # print(i+2, new_case[i], "1: ",mutation_rate[i], "\t :", new_case[i-1])
            # print(i + 2, new_case[i - 1], new_case[i], mutation_rate[i] )
            if (Hospitalized_data[i - 1] != 0):
                k = i - 1

    """
    
    data to show for prediction
    
    """

    mutation_df = pd.DataFrame(np.array(mutation_rate).reshape(-1, 1), columns=["Mutation"])

    mutation_df.to_csv(r"corona_data\\newdata\\total_mutation.csv", index=True)
    mutation_df = pd.read_csv("corona_data\\newdata\\total_mutation.csv")

    """
    
    
    Doing same for getting data of china 
    
    
    """

    Hospitalized_graph1 = Hospitalized_df["China"].fillna(method='pad')

    Hospitalized_data1 = []
    Hospitalized_data1.append(Hospitalized_graph1)
    Hospitalized_data1 = np.reshape(Hospitalized_data1, (1, -1))
    Hospitalized_data1 = list(chain.from_iterable(Hospitalized_data1))

    new_case1 = [int(Hospitalized_data1[0])]

    for i in range(1, len(Hospitalized_df)):
        new_case1.append(int(Hospitalized_data1[i]) - int(Hospitalized_data1[i - 1]))

    newcase_df1 = pd.DataFrame(np.array(new_case1).reshape(-1, 1), columns=["New"])

    k = 0
    j = 0

    """
    
    Mutation rate calculator
    
    """

    mutation_rate1 = [int(Hospitalized_data1[0])]
    # print("i k i i-1 rate")
    # print("2 ",new_case[0],"1: ", mutation_rate[0])
    for i in range(1, len(Hospitalized_data1)):
        # print(i, k, new_case[i], new_case[i - 1], mutation_rate[i - 1])
        if (Hospitalized_data1[i - 1] == 0 and Hospitalized_data1[i] == 0):
            mutation_rate1.append(0)
            # print(i+2, new_case[i], "0: ", mutation_rate[i])
        elif (Hospitalized_data1[i] != 0 and Hospitalized_data1[i - 1] == 0):
            if (k == 0):
                mutation_rate1.append(0)
            else:
                mutation_rate1.append(((Hospitalized_data1[i] / (Hospitalized_data1[k]))))
            # print(i+2, new_case[i], "2: ", mutation_rate[i])
        else:

            mutation_rate1.append(((Hospitalized_data1[i] / (Hospitalized_data1[i - 1]))))
            # print(i+2, new_case[i], "1: ",mutation_rate[i], "\t :", new_case[i-1])
            # print(i + 2, new_case[i - 1], new_case[i], mutation_rate[i] )
            if (Hospitalized_data1[i - 1] != 0):
                k = i - 1

    """
    
    mutaion to csv for china
    
    """

    mutation_df1 = pd.DataFrame(np.array(mutation_rate1).reshape(-1, 1), columns=["Mutation"])
    mutation_df1.to_csv(r"corona_data\\newdata\\total_mutation1.csv", index=True)
    mutation_df1 = pd.read_csv("corona_data\\newdata\\total_mutation1.csv")

    """
    Data calulation for Graph show
    """

    X = np.linspace(1, len(mutation_df["Mutation"]), len(mutation_df["Mutation"]))
    X_plot = np.linspace(0, 200, 15)

    """
    
    Data old graph show
    
    
    plt.figure(figsize=(12,6.5))
    plt.xticks(X_plot)
    plt.ylabel("Population affected")
    plt.xlabel("Days")
    l1 = plt.plot(X,Y)
    l2 = plt.plot(X,Y1)
    plt.show()
    
    
    
    plt.figure(figsize=(13,6.5))
    plt.title("Mutation Rate")
    plt.plot(X,Y2)
    plt.ylabel("Mutation Rate")
    plt.xlabel("Days")
    plt.xticks(X_show)
    # plt.show()
    
    
    """

    """
    
    prediction
    
    
    """
    Lockdown = pd.read_csv("corona_data\\newdata\\Lockdown.csv")
    Lockdown_pred = pd.read_csv("corona_data\\newdata\\Lockdownpred.csv")

    # taking data X_pred  is for train
    # taking data X_pred1  is for test
    # taking data Y_pred for prediction
    """
    
     if we remove lockdown prediction will be correct 
     but we should train and test data for same country
     otherwise score will not be good
     if we train and test for same country then lockdown is of no use
     i think we should change data of mutation on lockdown
    
    """

    Y_pred = newcase_df["New"]
    X_pred = pd.concat([mutation_df["Mutation"], Hospitalized_df[country].fillna(method='pad')], axis=1, sort=False)
    dateget = len(mutation_df)
    newdate = np.linspace(1, dateget, dateget)
    """
    Method
    
    
    """
    Model = linear_model.LinearRegression()
    Model.fit(X_pred, Y_pred)
    Prediction = Model.predict(X_pred)
    ent1['state'] = 'normal'
    ent1.delete(0, END)
    ent1.insert(0,"Prediction Score: " + str(int(Model.score(X_pred, Y_pred)*100)) + "%")
    ent1['state'] = 'disabled'

    """
    
    for taking sum of precdtion and sum of original to check wheater the output is correct or not 
    
    """
    sum_pred = []
    sm = 0
    for i in range(len(Prediction)):
        # sm = sm + Prediction[i]
        sm = Prediction[i]
        sum_pred.append(sm)
    Pred_df = pd.DataFrame(np.array(sum_pred).reshape(-1, 1), columns=["Prediction"])
    Pred_df.to_csv("corona_data\\newdata\\prediction.csv")

    # prediction Graph show
    plt.figure(figsize=(5.5, 5.5))
    plt.title("Data Of " + country)
    plt.xticks(X_plot)
    plt.ylabel("Population affected")
    plt.xlabel("Days")
    plt.plot(X, np.array(Prediction), "black", label="Prediction", alpha=0.5)
    plt.plot(X, Y_pred, "green", label="Original", alpha=0.5)
    plt.legend()
    plt.show()


ent = Entry(window, bg="#efefef", fg="#000000", bd=0, justify="center", font=("comic sans ms", "12"))
ent.pack(padx=20, pady=6)
ent.bind("<Return>", startsearch)
btn = Button(window, bg="#4885ef", fg="#ffffff", bd=0, text="start", command=lambda: startsearch(0),
             font=("comic sans ms", "11"), padx=20, pady=0)
btn.pack(padx=20, pady=6)

ent1 = Entry(window, bg="#ffffff", width=35, disabledbackground="#ffffff", fg="#cfcfcf", bd=0, justify="center",
             font=("comic sans ms", "12"), state="disabled")
ent1.pack(padx=20, pady=6)
window.mainloop()
