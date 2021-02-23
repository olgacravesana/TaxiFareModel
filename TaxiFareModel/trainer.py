# imports
from TaxiFareModel.encoders import TimeFeaturesEncoder,DistanceTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.utils import compute_rmse
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from TaxiFareModel.data import get_data, clean_data


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline= None
        self.X = X
        self.y = y

    # def fit(self):
    #     self.pipeline.fit(self.X,self.y)


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe= Pipeline([('distance', DistanceTransformer()),
                        ('scaler', StandardScaler())])
        time_pipe= Pipeline([('time_encode',TimeFeaturesEncoder('pickup_datetime')),
                        ('one_hot', OneHotEncoder())])
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']
        preprocessing = ColumnTransformer([('distance', dist_pipe,dist_cols),
                                            ('time', time_pipe,time_cols)])
        self.pipeline = Pipeline([('preprocessing', preprocessing),
                        ('model', LinearRegression())])
        # self.pipeline=pipeline

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X,self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return compute_rmse(y_pred,y_test)


if __name__ == "__main__":
    df = get_data()
    df_cleaned = clean_data(df)
    # set X and y
    X = df_cleaned.drop(columns = 'fare_amount')
    y = df_cleaned.fare_amount
    # hold out
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.1)
    # train
    trainer = Trainer(X_train,y_train)
    trainer.run()
    # evaluate
    rmse = trainer.evaluate(X_test, y_test)
    print('TODO')
