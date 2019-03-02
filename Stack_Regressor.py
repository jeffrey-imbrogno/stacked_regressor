    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR, LinearSVR
    from sklearn import linear_model
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    
    #This function requires a dataframe, the number of folds (default is 5), and the dependent variable. 
    def stackedregressor(df, y, fld=5):
        #The empty list will have a random integer for each row. 
        fldlst = []
        counter = 0
        c = 1
        rows = len(df.index)
        
        #This creates the list of random integers from 1 to the total number of folds desired
        while counter < rows:
            r = np.random.randint(1,fld)
            fldlst.append(r)
            counter += 1
        
        #This adds the fold number to the dataframe
        df['folds'] = fldlst
        
        #We begin a loop to run various regression algorithms on each fold. The output of each algorithm will be used to create another dataframe
        while c <= fld
            #Train-test split
            foldtrain = df[df.folds != c]
            foldtest = df[df.folds == c]
            
            #Creation of X,Y variables
            X_train = foldtrain.drop(y, axis=1)
            y_train = foldtrain[y]
            X_test = foldtest.drop(y, axis=1)
            y_test = foldtest[y]
            
            #Creation of a meta dataframe so the output of each regression will be properly indexed
            meta_df = foldtest
            
            #Linear Regression
            linreg = LinearRegression()
            linreg.fit(X_train, y_train)
            y_pred1 = linreg.predict(X_test)
            meta_df['linreg'] = y_pred1
            
            #SVR
            svr = SVR()
            svr.fit(X_train, y_train)
            y_pred2 = svr.predict(X_test)
            meta_df['svr'] = y_pred2
            
            #Linear SVR
            linear_svr = LinearSVR()
            linear_svr.fit(X_train, y_train)
            y_pred3 = linear_svr.predict(X_test1)
            meta_df['linsvr'] = y_pred3
            
            #Random forest regressor
            rfr= RandomForestRegressor(n_estimators=300)
            rfr.fit(X_train, y_train)
            y_pred4 = rfr.predict(X_test)
            meta_df['rfr'] = y_pred4
            
            #Lasso regression
            lasso = linear_model.Lasso()
            lasso.fit(X_train, y_train)
            y_pred5 = lasso.predict(X_test)
            meta_df['lasso'] = y_pred5
            
            #Gradient Boosting Regressor
            gbr = GradientBoostingRegressor()
            gbr.fit(X_train, y_train)
            y_pred6 = gbr.predict(X_test)
            meta_df['gbr'] = y_pred6
            
            #Extra Trees Regressor
            etr = ExtraTreesRegressor()
            etr.fit(X_train, y_train)
            y_pred7 = etr.predict(X_test)
            meta_df['etr'] = y_pred7
            
            #After the meta_df for the fold is created, the initial x variables will need to be dropped
            droplist = list(X_test)
            meta_df = meta_df.drop(droplist, axis=1)
            #The output dataframe should be only the results of each regression and the y variable.
            output_df = output_df.join(meta_df)
            
            c += 1
        
        return output_df