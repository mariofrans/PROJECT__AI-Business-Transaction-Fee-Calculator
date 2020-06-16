import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def main():
    print("Artificial Intelligence Transaction Fee Calculator")
    print("==================================================================")

    # user input filename which has the previous data to be used
    filename = str(input("Please input the file name which has previous transaction data/s [amount & fee]: "))

    # read csv file
    file = pd.read_csv(filename)
    print("File content: ")
    print(file)

    # input file data into arrays
    amount = file.Amount
    fee = file.Fee

    # convert these array into numpy so that it can be read
    x = np.array(amount).reshape(-1, 1)
    y = np.array(fee)

    # create linear regression model of these values
    model = LinearRegression()
    model.fit(x, y)

    # calculate and print the y-intercept, gradient, and equation of line with the data collected
    print("==================================================================")
    print("Fixed Fee [y-intercept]: ", float(model.intercept_))
    print("Proportional Fee per Transaction Amount [gradient]: ", float(model.coef_))
    print("Transaction Fee [equation]= ", float(model.coef_), "* (Transaction Amount) +", float(model.intercept_))
    print("==================================================================")

    # input the number of future transactions fee/s to be predicted
    n = int(input("Please enter the number of future transactions fee/s to be predicted: "))

    # create empty array for prediction
    amount_pred = []
    fee_pred = []

    # input & predict the transaction amount & fee respectively
    for i in range(n):
        amount_pred.append(float(input("transaction amount = ")))
        fee_pred.append((model.coef_ * amount_pred[i]) + model.intercept_)
    print("==================================================================")

    for i in range(n):
        print("The predicted transaction fee for ",amount_pred[i]," is: ", float(fee_pred[i]))
    print("==================================================================")

    # combine the previous data and the predicted data
    combine_x = np.concatenate((amount, amount_pred), axis=None)
    combine_y = np.concatenate((fee, fee_pred), axis=None)

    # sort the combined arrays
    combine_x.sort()
    combine_y.sort()

    # combine the 2 arrays into a double array
    df = np.column_stack((combine_x, combine_y))
    # print(df)

    # override the csv file data
    np.savetxt('transactions.csv', df, fmt='%.2f', delimiter=',', header="Amount,Fee", comments='')

    # show updated file content
    file = pd.read_csv(filename)
    print("Updated file content: ")
    print(file)

    # ask user if they want to repeat the program
    print("==================================================================")
    print("Do you want to repeat the program? Please choose an option")
    print("1. Yes")
    print("2. No")
    option = int(input("Input: "))
    print("==================================================================")

    if(option == 1):
        main()
    elif(option == 2):
        print("Thank you for using the program!")

main()


# transactions.csv

# Amount,Fee
# 19.99,1.18
# 29.99,1.62
# 39.98,2.06
# 52.98,2.63